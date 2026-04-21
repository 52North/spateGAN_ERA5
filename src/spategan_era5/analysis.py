# Add project root to path
import sys



import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import itertools
sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.animation as animation
from src.spategan_era5.utils import _combine_datasets
from src.spategan_era5.plotting import Plots

def visualize_precipitation_map(combined_ds, time_index=4):
    # Select time slice
    ds_time = combined_ds.isel(time=time_index)

    # Convert to DataArray to facilitate faceting
    plot_data = ds_time[["era5_tp", "pred_precipitation"]].to_array()

    # Use xarray's native plotting with faceting
    g = plot_data.plot(
        col="variable",
        cmap="YlGnBu",
        figsize=(14, 6),
        subplot_kws={"title": ["Ground Truth (ERA5 Interpolated)", "Prediction (SpateGAN)"]}
    )

    # Adjust title
    time_str = ds_time.time.dt.strftime("%Y-%m-%d %H:%M").values
    g.fig.suptitle(f"Precipitation Comparison at {time_str}", y=1.05)

    plt.show()


def check_data_health(combined_ds, time_index=0):
    ds_time = combined_ds.isel(time=time_index)

    print(f"--- Time: {ds_time.time.values} ---")

    # Check Ground Truth (ERA5)
    era5_min = float(ds_time["era5_tp"].min())
    era5_max = float(ds_time["era5_tp"].max())
    print(f"ERA5 min/max: {era5_min:.6f} to {era5_max:.6f}")

    # Check Prediction (SpateGAN)
    pred_min = float(ds_time["pred_precipitation"].min())
    pred_max = float(ds_time["pred_precipitation"].max())
    pred_nans = int(ds_time["pred_precipitation"].isnull().sum())

    print(f"SpateGAN min/max: {pred_min:.6f} to {pred_max:.6f}")
    print(f"SpateGAN NaN count: {pred_nans}")


def debug_spategan_nans(spategan_path: Path | str, combined_ds: xr.Dataset):
    ds_raw = xr.open_dataset(spategan_path)
    raw_total_nans = int(ds_raw["precipitation"].isnull().sum())
    raw_total_valid = int(ds_raw["precipitation"].notnull().sum())

    print(f"Raw SpateGAN file - Valid values: {raw_total_valid}, NaNs: {raw_total_nans}")

    valid_pixels_per_time = combined_ds["pred_precipitation"].notnull().sum(dim=["lat", "lon"])

    # Extract all indices where the valid pixel count is greater than zero
    valid_indices = np.where(valid_pixels_per_time.values > 0)[0]

    if len(valid_indices) == 0:
        print("CRITICAL: The merged dataset contains 100% NaNs for SpateGAN across all times.")
        print("This means `xr.align` failed to match the exact timestamps. You may need to round the datetime coordinates before merging.")
    else:
        print(f"Found {len(valid_indices)} valid timesteps:")
        for idx in valid_indices:
            timestamp = combined_ds.time[idx].values
            pixel_count = int(valid_pixels_per_time[idx].values)
            print(f"  - Index: {idx:3d} | Timestamp: {timestamp} | Valid pixels: {pixel_count}")


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


if __name__ == "__main__":
    era5_file = Path("data/era5_2026-03-01_to_2026-03-04_clean.nc")
    spategan_file = Path("output/utm/spateGAN_ERA5_latlon_52.12N_6.88E_20260301_20260303_e10.nc")

    if era5_file.exists() and spategan_file.exists():

        spategan_ds = xr.open_dataset(spategan_file)
        era5_ds = xr.open_dataset(era5_file)
        combined_dataset = _combine_datasets(era5_ds, spategan_ds)
        visualize_precipitation_map(combined_dataset)

        # Call this before your visualization function
        check_data_health(combined_dataset, time_index=0)
        debug_spategan_nans(spategan_file, combined_dataset)

        # 1. Define your configuration and paths
        project_root = Path.cwd()
        config = {
            "data": {
                "plots_path": "plots/animations"
            },
            "processing": {
                "seed": 42  # Add this block
            }
        }



        # 2. Instantiate the Plots class with your loaded datasets
        plotter = Plots(
            predictions_utm=spategan_ds, # Your loaded prediction dataset
            ds_utm_28=era5_ds,             # Your loaded ground truth dataset
            project_root=project_root,
            config=config
        )

        # 3. Run the animation
        # To save as a file:
        plotter.create_precipitation_animation(num_frames=24, save_filename="precip_comparison.mp4")
    else:
        print("Dataset paths do not exist. Please check the file locations.")
