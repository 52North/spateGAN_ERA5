import logging
from pathlib import Path
import xarray as xr
from ecmwf.opendata import Client # Copernicus Data Store API client
import logging
import pandas as pd
import cdsapi
import itertools
import yaml
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def prepare_ecmwf_data(out_dir: Path, date: str, forecast_steps: list) -> Path:
    """Download and prepare AIFS data, with fallback to historical MARS archive."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_filename = out_dir / f"ecmwf_{date}_aifs.grib2"
    clean_filename = out_dir / f"ecmwf_{date}_clean.nc"

    if clean_filename.exists():
        logger.info(f"Skipping: Processed file {clean_filename} already exists.")
        return clean_filename

    if raw_filename.exists():
        logger.info(f"Skipping download: Raw file {raw_filename} already exists.")
    else:

        logger.info(f"Downloading AIFS data for date '{date}' and steps '{forecast_steps}'")
        client = Client(source="azure", model="aifs-single")
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

    if ds['tp'].attrs.get('units') == 'kg m**-2':
        ds_mm = ds_hourly_rate / 1000
    else:
        ds_mm = ds_hourly_rate

    ds_spateGAN = ds_mm[['cp', 'lsp']].clip(min=0)
    ds_spateGAN.to_netcdf(clean_filename)

    return clean_filename

def download_era5(out_dir: Path, start_date: str, end_date: str) -> Path:
    """Download historical ERA5 data from Copernicus CDS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_filename = out_dir / f"era5_{start_date}_to_{end_date}.nc"

    if raw_filename.exists():
        logger.info(f"Skipping download: {raw_filename} already exists.")
        return raw_filename

    logger.info(f"Downloading ERA5 data from '{start_date}' to '{end_date}'")
    client = cdsapi.Client()

    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'convective_precipitation',
                'total_precipitation'
            ],
            'date': f"{start_date}/{end_date}",
            'time': [f"{i:02d}:00" for i in range(24)],
            'format': 'netcdf',
        },
        str(raw_filename)
    )

    return raw_filename

def prepare_era5(out_dir: Path, start_date: str, end_date: str) -> Path:
    """Download and prepare ERA5 data to match AIFS structure."""
    raw_filename = download_era5(out_dir, start_date, end_date)
    clean_filename = out_dir / f"era5_{start_date}_to_{end_date}_clean.nc"

    if clean_filename.exists():
        return clean_filename

    ds = xr.open_dataset(raw_filename)

    # Match dimensions
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})

    # Match variables: calculate Large-scale precipitation (lsp) to match AIFS
    if 'tp' in ds.data_vars and 'cp' in ds.data_vars:
        ds['lsp'] = ds['tp'] - ds['cp']

    ds.to_netcdf(clean_filename)
    return clean_filename

def compare_dataset_structures(file_path_1: Path, file_path_2: Path) -> dict:
    """Compare variables and dimensions of two datasets and print a formatted table."""
    engine1 = "cfgrib" if str(file_path_1).endswith(".grib2") else None
    engine2 = "cfgrib" if str(file_path_2).endswith(".grib2") else None

    ds1 = xr.open_dataset(file_path_1, engine=engine1)
    ds2 = xr.open_dataset(file_path_2, engine=engine2)

    v_match = list(set(ds1.data_vars).intersection(ds2.data_vars))
    v_only1 = list(set(ds1.data_vars) - set(ds2.data_vars))
    v_only2 = list(set(ds2.data_vars) - set(ds1.data_vars))

    d_match = list(set(ds1.dims).intersection(ds2.dims))
    d_only1 = list(set(ds1.dims) - set(ds2.dims))
    d_only2 = list(set(ds2.dims) - set(ds1.dims))

    n1, n2 = file_path_1.name, file_path_2.name
    w = max(len(n1), len(n2), 15)

    def build_table(title, only1, match, only2):
        lines = [
            f"\n{title:^{w*3 + 6}}",
            f"{n1:^{w}} | {'Common':^{w}} | {n2:^{w}}",
            "-" * (w * 3 + 6)
        ]
        for row in itertools.zip_longest(only1, match, only2, fillvalue=""):
            lines.append(f"{row[0]:^{w}} | {row[1]:^{w}} | {row[2]:^{w}}")
        lines.append("-" * (w * 3 + 6))
        return "\n".join(lines)

    # Using print to avoid logger prefixes breaking the table alignment
    print(build_table(" VARIABLES ", v_only1, v_match, v_only2))
    print(build_table(" DIMENSIONS ", d_only1, d_match, d_only2))

    return {
        "matching_variables": v_match,
        "variables_only_in_ds1": v_only1,
        "variables_only_in_ds2": v_only2,
        "matching_dimensions": d_match,
        "dimensions_only_in_ds1": d_only1,
        "dimensions_only_in_ds2": d_only2,
    }

def main():
    logging.basicConfig(level=logging.INFO)

    with open("config/config.yml", "r") as f:
        full_config = yaml.safe_load(f)

    config = full_config.get("data", {})

    out_dir_aifs = Path(config.get("aifs_out_dir", "./data/aifs"))
    out_dir_era = Path("./data")

    forecast_date_str = config.get("forecast_date")
    if forecast_date_str:
        aifs_date_dt = datetime.strptime(forecast_date_str, "%Y-%m-%d")
    else:
        aifs_date_dt = datetime.now()

    aifs_date = aifs_date_dt.strftime("%Y-%m-%d")

    forecast_stop = config.get("forecast_stop", 72)
    forecast_steps = list(range(0, forecast_stop + 1, 6))

    era5_start_dt = aifs_date_dt
    era5_end_dt = aifs_date_dt + timedelta(hours=forecast_stop)

    era5_start = era5_start_dt.strftime("%Y-%m-%d")
    era5_end = era5_end_dt.strftime("%Y-%m-%d")

    aifs_path = prepare_ecmwf_data(out_dir_aifs, aifs_date, forecast_steps)
    era5_path = prepare_era5(out_dir_era, era5_start, era5_end)

    comparison = compare_dataset_structures(aifs_path, era5_path)
    return comparison

if __name__ == "__main__":
    main()
