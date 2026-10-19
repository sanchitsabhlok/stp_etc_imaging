import importlib.metadata
from pathlib import Path
__version__ = importlib.metadata.version(__package__ or "stp_etc_imaging")

__all__ = [ "__version__", "get_data_path"]

def get_data_path():
    package_root = Path(__file__).parent.resolve()
    data_path = package_root / "support_data"

    if not data_path.exists():
        raise FileNotFoundError(f"Support data directory not found: {data_path}")

    return str(data_path) + "/"
