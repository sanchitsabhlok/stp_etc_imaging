import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np
from math import ceil
import astropy.units as u
from synphot import SpectralElement
from synphot.models import Box1D

try:
    import config_stp
    import config_stp_wcc
    import config_um
    import config_um_wcc
except:
    print("Could not import configuration repos.")

try:
    import utils_config
except:
    print("Could not import utils_config. Cannot parse tomls. Exiting.")
    exit()

from stp_etc_imaging import ExposureTimeSNRCalculator as etsc

@pytest.fixture
def UM():
    return "UM"

@pytest.fixture
def STP():
    return "STP"

@pytest.fixture
def test_STP():
    return "test_STP"

@pytest.fixture
def test_UM():
    return "test_UM"

@pytest.fixture
def telescope(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("telescope", ["UM", "STP", "test_STP"], indirect=True)
def test_configs_telescope(telescope):
    if telescope == "STP":
        data_telescope = config_stp.load_config_values("parsed")
    if telescope == "UM":
        data_telescope = config_um.load_config_values("parsed")
    if telescope == "test_STP":
        test_loader = utils_config.ConfigLoader('test_data/test_STP', "parsed")
        data_telescope = test_loader.load_configs()
    if telescope == "test_UM":
        test_loader = utils_config.ConfigLoader('test_data/test_UM', "parsed")
        data_telescope = test_loader.load_configs()

    assert "value" in data_telescope['observatory']['pointing']['jitter_rms'], f"data_telescope['observatory']['pointing']['jitter_rms'] could not be verified in {telescope} config"
    assert "value" in data_telescope['telescope']['optics']['m1']['aper_clear_OD'], f"data_telescope['telescope']['optics']['m1']['aper_clear_OD'] could not be verified in {telescope} config"
    assert "value" in data_telescope['telescope']['optics']['m1']['surface_rms'], f"data_telescope['telescope']['optics']['m1']['surface_rms'] could not be verified in {telescope} config"
    assert isinstance(data_telescope['telescope']['general']['f_number'], float), f"data_telescope['telescope']['general']['f_number'] could not be verified in {telescope} config"
    assert "value" in data_telescope['telescope']['optics']['m2']['aper_clear_OD'], f"data_telescope['telescope']['optics']['m2']['aper_clear_OD'] could not be verified in {telescope} config"
    assert "value" in data_telescope['telescope']['optics']['m2']['support_width'], f"data_telescope['telescope']['optics']['m2']['support_width'] could not be verified in {telescope} config"
    assert isinstance(data_telescope['telescope']['optics']['m2']['n_supports'], int), f"data_telescope['telescope']['optics']['m2']['n_supports'] could not be verified in {telescope} config"

    return

@pytest.mark.parametrize("telescope", ["UM", "STP"], indirect=True)
def test_configs_instrument(telescope):
    if telescope == "STP":
        data_instrument = config_stp_wcc.load_config_values("parsed")
    if telescope == "UM":
        data_instrument = config_um_wcc.load_config_values("parsed")

    #   Sensor table is either [sensor] or [sensor.<sensor_name>]
    data_sensor = data_instrument['common_params']['sensor']
    if 'sensor_name' in data_sensor:
        assert data_sensor['sensor_name'] in data_sensor, f"Sensor {data_sensor['sensor_name']} not found in {telescope} instrument config"
        data_sensor = data_sensor[data_sensor['sensor_name']]

    assert "value" in data_sensor['temp_nominal'], "Value type mismatch. Potentially corrupted or modified data."
    assert isinstance(data_sensor['gain'], int), "Value type mismatch. Potentially corrupted or modified data."
    assert "value" in data_sensor['pixel_size'], "Value type mismatch. Potentially corrupted or modified data."

    return

@pytest.mark.parametrize("telescope", ["UM", "STP", "test_STP", "test_UM"], indirect=True)
def test_default_throughput(telescope):
    obs = etsc.Observatory(telescope)
    obs.make_STP()
    wave = obs.bandpass.waveset
    flux = obs.bandpass(wave)
    waveset_non_zero = wave[flux!=0]

    #   Read the default filter independently of the ETC, from the same file the ETC loaded.
    #   The ETC pads the filter with zero throughput on either end, so compare against the padded filter.
    filter_bp = obs.crop_spectral_element(SpectralElement.from_file(obs.filters[0].split(' x ')[0]))

    #   The non zero throughput must lie within the range of the filter.
    assert np.min(waveset_non_zero) >= np.min(filter_bp.waveset), "Throughput check failed. Throughput found blueward of the default filter."
    assert np.max(waveset_non_zero) <= np.max(filter_bp.waveset), "Throughput check failed. Throughput found redward of the default filter."
    #   Throughput is physical, and can never exceed the filter transmission.
    assert np.all(flux >= 0) and np.all(flux <= 1), "Throughput check failed. Throughput outside [0, 1]."
    assert np.all(flux.value <= filter_bp(wave).value + 1e-9), "Throughput check failed. Throughput exceeds the filter transmission."
    #   The total throughput is the product of the telescope and instrument throughputs.
    assert np.allclose(flux, obs.telescope_bandpass(wave) * obs.instrument_bandpass(wave), rtol=1e-6), "Throughput check failed. Total throughput is not the product of telescope and instrument throughputs."

    return

@pytest.mark.parametrize("telescope", ["UM", "STP"], indirect=True)
def test_sensor_initialization(telescope):
    if telescope == "STP":
        data_telescope = config_stp.load_config_values("parsed")
        data_instrument = config_stp_wcc.load_config_values("parsed")
    if telescope == "UM":
        data_telescope = config_um.load_config_values("parsed")
        data_instrument = config_um_wcc.load_config_values("parsed")
    data_sensor = data_instrument['common_params']['sensor']
    if 'sensor_name' in data_sensor:
        data_sensor = data_sensor[data_sensor['sensor_name']]

    obs = etsc.Observatory(telescope)
    obs.make_STP()

    #   Independent plate scale [arcsecond/pixel] = pixel size / (telescope diameter * f#) * 206265 arcseconds/radian
    pixel_size = (data_sensor['pixel_size']['value'] * u.Unit(data_sensor['pixel_size']['unit'])).to(u.um)
    diameter = (data_telescope['telescope']['optics']['m1']['aper_clear_OD']['value'] * u.Unit(data_telescope['telescope']['optics']['m1']['aper_clear_OD']['unit'])).to(u.um)
    f_num = data_telescope['telescope']['general']['f_number']
    expected_plate_scale = (pixel_size / (diameter * f_num)).value * 206265
    assert np.isclose(obs.plate_scale, expected_plate_scale, rtol=1e-6), "Sensor check: Incorrect plate scale. Potentially corrupt or modified configuration file."

    #   Independent PSF size: Airy disk diameter (2 * 1.22 * lambda * f#) at the peak wavelength, as an n x n square of pixels
    psf_diameter = 2 * 1.22 * obs.bandpass.wpeak().to(u.um) * f_num
    n = ceil((psf_diameter / pixel_size).value)
    assert obs.num_psf_pixels.value == n**2, "Incorrect PSF size. Indicates unexpected change in input parameters."

    return

@pytest.mark.parametrize("telescope", ["UM", "test_UM", "STP", "test_STP"], indirect=True)
def test_counts(telescope):
    if telescope == "STP":
        data_telescope = config_stp.load_config_values("parsed")
        data_path_telescope = config_stp.get_data_path()
    if telescope == "UM":
        data_telescope = config_um.load_config_values("parsed")
        data_path_telescope = config_um.get_data_path()
    if telescope == "test_STP":
        test_loader = utils_config.ConfigLoader('test_data/test_STP', "parsed")
        data_telescope = test_loader.load_configs()
        data_path_telescope = config_stp.get_data_path()
    if telescope == "test_UM":
        test_loader = utils_config.ConfigLoader('test_data/test_UM', "parsed")
        data_telescope = test_loader.load_configs()
        data_path_telescope = config_um.get_data_path()

    obs = etsc.Observatory(telescope)
    obs.make_STP()
    obs.set_source(source_pickles_file='test_data/pickles_uk_9.fits',
                   plot=True)
    obs.set_background(background_file=data_telescope['astrophysics']['zodi']['profile'], support_data_path=data_path_telescope,
                       plot=True)
    obs.make_observation(flux=0.0, flux_units='AB')

    #   End to end checks without hard coded truth values. Each ETC output is checked against an independent
    #   relation it must satisfy, so updates to constituent files (QE curves, mirrors, filters, tomls) do not
    #   require updating the test, while broken code still fails.
    source_counts_zero_mag = obs.source_counts.value
    sky_counts = obs.sky_counts.value
    assert source_counts_zero_mag > 0, "Source counts must be positive."
    assert sky_counts > 0, "Sky counts must be positive."

    #   Source counts scale as 10**(-0.4 * magnitude). Sky counts are independent of the source flux.
    test_flux_mag = 20.0
    obs.make_observation(flux=test_flux_mag, flux_units='AB')
    assert np.isclose(obs.source_counts.value, source_counts_zero_mag * 10**(-0.4 * test_flux_mag), rtol=1e-6), "Source counts do not scale with source magnitude."
    assert np.isclose(obs.sky_counts.value, sky_counts, rtol=1e-9), "Sky counts changed with source magnitude."

    ee = obs.encircled_energy
    source_counts = obs.source_counts.value
    dc = obs.dark_current.value
    rn = obs.read_noise.value
    n_psf = obs.num_psf_pixels.value
    int_time = 1.0
    exp_time = 1.0

    #   SNR against the CCD equation evaluated independently
    snr_ccd = lambda s, t, t_exp: ee * s * t / np.sqrt(ee * s * t + sky_counts * t + (dc + rn * rn / t_exp) * t * n_psf)
    assert np.isclose(obs.calc_SNR(int_time=int_time * u.s, exp_time=exp_time * u.s), snr_ccd(source_counts, int_time, exp_time), rtol=1e-6), "SNR does not match the CCD equation."

    #   Integration time for a target SNR must give back the target SNR
    target_snr = 100.0
    req_int_time = obs.calc_int_time(target_snr, exp_time=exp_time * u.s)
    assert np.isclose(obs.calc_SNR(int_time=req_int_time, exp_time=exp_time * u.s), target_snr, rtol=1e-6), "Integration time does not give back the target SNR."

    #   Required source counts and magnitude for a target SNR must give back the target SNR
    target_snr = 10.0
    req_source = obs.calc_req_source(target_snr, int_time=int_time * u.s, exp_time=exp_time * u.s)
    assert np.isclose(snr_ccd(req_source, int_time, exp_time), target_snr, rtol=1e-6), "Required source counts do not give back the target SNR."
    req_mag = obs.calc_req_source(target_snr, int_time=int_time * u.s, exp_time=exp_time * u.s, magnitude=True)[0]
    obs.make_observation(flux=float(np.squeeze(req_mag)), flux_units='AB')
    assert np.isclose(obs.source_counts.value, req_source, rtol=1e-6), "Required magnitude does not give back the required source counts."
    assert np.isclose(obs.calc_SNR(int_time=int_time * u.s, exp_time=exp_time * u.s), target_snr, rtol=1e-6), "Required magnitude does not give back the target SNR."

    #   At the saturation time a PSF pixel has collected exactly the well depth
    pixel_rate = (ee * obs.source_counts.value + sky_counts) / n_psf
    assert np.isclose(obs.calc_saturation_time().value * pixel_rate, obs.well_depth.value, rtol=1e-6), "Saturation time does not fill the well depth."
    return

@pytest.mark.parametrize("telescope", ["UM", "STP", "test_UM", "test_STP"], indirect=True)
def test_validate_ETC_snr_calculation(telescope):
    if telescope == "STP":
        data_telescope = config_stp.load_config_values("parsed")
        data_path_telescope = config_stp.get_data_path()
    if telescope == "UM":
        data_telescope = config_um.load_config_values("parsed")
        data_path_telescope = config_um.get_data_path()
    if telescope == "test_STP":
        test_loader = utils_config.ConfigLoader('test_data/test_STP', "parsed")
        data_telescope = test_loader.load_configs()
        data_path_telescope = config_stp.get_data_path()
    if telescope == "test_UM":
        test_loader = utils_config.ConfigLoader('test_data/test_UM', "parsed")
        data_telescope = test_loader.load_configs()
        data_path_telescope = config_um.get_data_path()

    zodi_magnitude_normalization = float(data_telescope['astrophysics']['zodi']['zodi_mag_r'])
    obs = etsc.Observatory(telescope)
    obs.make_STP(mirrors=False, filters=False)
    obs.bandpass *= SpectralElement(Box1D, amplitude=1, x_0=5500, width=1)

    test_flux_mag = 18.0
    test_wavelength = 5500 #    Angstroms
    obs.set_source(source_pickles_file='test_data/pickles_uk_9.fits',
                   plot=True)
    obs.set_background(background_file=data_telescope['astrophysics']['zodi']['profile'], support_data_path=data_path_telescope,
                       plot=True)
    obs.make_observation(flux=test_flux_mag)

    bandpass_val = obs.bandpass(test_wavelength)

    #   Run independent calculations of expected source and sky counts
    #   Source counts
    flux_source = test_flux_mag * u.ABmag
    flambda_flux_source = flux_source.to(u.erg / u.s / u.cm ** 2 / u.AA, equivalencies=u.spectral_density(test_wavelength * u.AA))
    flambda_flux_photon_source = flambda_flux_source / 6.626e-27 / 299792458 * test_wavelength * 1e-10 * u.photon / u.erg

    flambda_flux_photon_telescope_area_source = flambda_flux_photon_source * (obs.surf_area.value*10000) * u.cm ** 2 * u.AA  # For a 1 Angstrom narrowband filter
    flambda_flux_photon_telescope_area_bandpass_source = flambda_flux_photon_telescope_area_source * bandpass_val

    #   Sky counts
    flux_sky = obs.bg_magnitude * u.ABmag
    flambda_flux_sky = flux_sky.to(u.erg / u.s / u.cm ** 2 / u.AA, equivalencies=u.spectral_density(test_wavelength * u.AA))
    flambda_flux_photon_sky = flambda_flux_sky / 6.626e-27 / 299792458 * test_wavelength * 1e-10 * u.photon / u.erg

    flambda_flux_photon_telescope_area_sky = flambda_flux_photon_sky * (obs.surf_area.value*10000) * u.cm ** 2 * u.AA  # For a 1 Angstrom narrowband filter
    flambda_flux_photon_telescope_area_bandpass_sky = flambda_flux_photon_telescope_area_sky * bandpass_val

    #   Zodi is normalized in Johnson V band
    #   Vega has V mag = 0.03 in Johnson V band
    #   Johnson V band mag = AB mag
    #   https://lweb.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html
    #   V band to AB conversion => V = M(AB) + 0.03
    #   0.03 = V-M(AB) = - 2.5 * log10(F_V/F_AB)
    #   => 10 ** (0.03/-2.5) * F_AB = F_V
    #   This seems to be the root cause of the issue, but can't confirm that the Zodi isn't in V magnitudes
    #   flambda_flux_photon_telescope_area_bandpass_sky_V = flambda_flux_photon_telescope_area_bandpass_sky / 10**(-0.044/2.5)

    #   Check the source count calculation from independent check is good to within 0.01 mag
    assert abs((-2.5 * np.log10(obs.source_counts.value/flambda_flux_photon_telescope_area_bandpass_source.value))) < 0.01
    #   Check the sky count calculation from independent check is good to within 0.05 mag
    assert abs((-2.5 * np.log10(obs.sky_counts.value/flambda_flux_photon_telescope_area_bandpass_sky.value))) < 0.05

    #   Check the actual SNR calculation

    dc = obs.dark_current.value
    rn = obs.read_noise.value
    n_psf = obs.num_psf_pixels.value
    int_time = 100.0
    exp_time = 100.0

    ee = obs.encircled_energy

    total_noise = np.sqrt(ee * flambda_flux_photon_telescope_area_bandpass_source.value * int_time + flambda_flux_photon_telescope_area_bandpass_sky.value * int_time
                       + (dc + rn * rn / exp_time) * int_time * n_psf)
    snr = ee * flambda_flux_photon_telescope_area_bandpass_source.value * int_time / total_noise
    snr_etc = obs.calc_SNR(int_time*u.s, exp_time*u.s)

    #   Check SNR is good to within 0.01 %
    assert abs(1-snr_etc/snr)*100.0 < 0.01

    return


def test_get_interpolated_value(tmp_path):
    #   Unit test on a synthetic file, independent of any sensor data
    input_file = tmp_path / "curve.csv"
    input_file.write_text("x,y\n0,0\n10,20\n20,60\n")
    obs = etsc.Observatory("STP")

    assert np.isclose(obs.get_interpolated_value(input_file, 10, ['x', 'y']), 20.0), "Interpolation does not return tabulated value."
    assert np.isclose(obs.get_interpolated_value(input_file, 5, ['x', 'y']), 10.0), "Interpolation is not linear between tabulated values."
    assert np.isclose(obs.get_interpolated_value(input_file, 15, ['x', 'y']), 40.0), "Interpolation is not linear between tabulated values."

    return
