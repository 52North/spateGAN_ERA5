from pathlib import Path

import xarray as xr
import yaml

import matplotlib
matplotlib.use("Agg")
from spategan_era5.plotting import Plots


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

PROJECT_ROOT = Path('/work/PROJEKTE/VELOCITYADAPT/code/spateGAN')

predictions_utm = xr.open_dataset("output/utm/spateGAN_ERA5_utm_52.12N_6.88E_20260327_20260405_e10.nc")
ds_utm_28 = xr.open_dataset("output/utm/era5_utm_52.12N_6.88E_20260327_20260405.nc")
config = load_config(Path("config/ecwmf_aifs.yml"))

plot = Plots(predictions_utm, ds_utm_28, PROJECT_ROOT, con