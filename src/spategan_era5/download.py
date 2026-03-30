import logging
from pathlib import Path

import xarray as xr
from ecmwf.opendata import Client

logger = logging.getLogger(__name__)

def prepare_ecmwf_data(out_dir: Path, date: str, forecast_steps: list) -> Path:
    """Download and prepare AIFS data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_filename = out_dir / f"ecmwf_{date}_aifs.grib2"
    clean_filename = out_dir / f"ecmwf_{date}_clean.nc"

    logger.info(f"Downloading AIFS data for date '{date}' and steps '{forecast_steps}'")
    client = Client(source="ecmwf", model="aifs-single")
    client.retrieve(
            date=date,
            time=0,
            step=forecast_steps,
            type="fc",
            param=["cp", "tp"],
            target=str(raw_filename),
        )

    ds = xr.open_dataset(raw_filename, engine="cfgrib")
    ds = ds.rename({
        'valid_time': 'time',
        'time': 'start_time',
    })
    ds = ds.swap_dims({'step': 'time'})
    ds['lsp'] = ds['tp'] - ds['cp']

    ds_hourly = ds.resample(time="1h").interpolate(kind='linear')
    ds_hourly_rate = (ds_hourly.shift(time=-1) - ds_hourly).dropna(dim="time")

    logger.info(f"Data Units: '{ds['tp'].attrs.get('units')}'")

    if ds['tp'].attrs.get('units') == 'kg m**-2':
        ds_mm = ds_hourly_rate / 1000
    else:
        ds_mm = ds_hourly_rate

    ds_spateGAN = ds_mm[['cp', 'lsp']].clip(min=0)
    ds_spateGAN.to_netcdf(clean_filename)

    return clean_filename

#
# ds = xr.open_dataset("/work/PROJEKTE/VELOCITYADAPT/code/spateGAN/data/aifs/ecmwf_20260330_aifs.grib2")
# ds = ds.rename({
#     'valid_time': 'time',
#     'time': 'start_time',
# })
# ds = ds.swap_dims({'step': 'time'})
# ds['lsp'] = ds['tp'] - ds['cp']
#
# ds_hourly = ds.resample(time="1h").interpolate(kind='linear')
# ds_hourly_rate2 = (ds_hourly.shift(time=-1) - ds_hourly).dropna(dim="time")
# ds_hourly_rate = ds_hourly.diff(dim="time")