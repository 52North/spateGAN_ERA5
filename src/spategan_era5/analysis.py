import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import matplotlib.animation as animation

def visualize_matched_datasets(era5_path: Path | str, spategan_path: Path | str) -> xr.Dataset:
    ds_era5 = xr.open_dataset(era5_path)
    ds_spategan = xr.open_dataset(spategan_path)

    # 1. Standardize ERA5 coordinate names
    ds_era5 = ds_era5.rename({"latitude": "lat", "longitude": "lon"})
    ds_era5 = ds_era5.rename_vars({var: f"era5_{var}" for var in ds_era5.data_vars})
    ds_spategan = ds_spategan.rename_vars({"precipitation": "pred_precipitation"})

    # 2. Resample SpateGAN to hourly to match ERA5 time resolution
    # This sums the six 10-minute intervals within each hour into a single hourly value
    ds_spategan_hourly = ds_spategan.resample(time="1h").sum(skipna=True)

    # 3. Spatially interpolate ERA5 to match SpateGAN's grid
    ds_era5_spatial = ds_era5.interp(
        lat=ds_spategan_hourly.lat,
        lon=ds_spategan_hourly.lon,
        method="nearest"
    )

    # 4. Align by time to keep only overlapping hourly timestamps
    ds_era5_aligned, ds_spategan_aligned = xr.align(
        ds_era5_spatial,
        ds_spategan_hourly,
        join="inner"
    )

    # 5. Merge into a single dataset
    return xr.merge([ds_era5_aligned, ds_spategan_aligned])

import matplotlib.pyplot as plt

def visualize_precipitation_map(combined_ds, time_index=4):
    # Select a single time slice to visualize
    ds_time = combined_ds.isel(time=time_index)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot ERA5 ground truth (Total Precipitation 'tp')
    ds_time["era5_tp"].plot(
        ax=axes[0],
        cmap="YlGnBu",
        cbar_kwargs={"label": "Precipitation (ERA5)"}
    )
    axes[0].set_title("Ground Truth (ERA5 Interpolated)")

    # Plot SpateGAN prediction
    ds_time["pred_precipitation"].plot(
        ax=axes[1],
        cmap="YlGnBu",
        cbar_kwargs={"label": "Precipitation (SpateGAN)"}
    )
    axes[1].set_title("Prediction (SpateGAN)")

    # Add a main title with the current timestamp
    time_str = ds_time.time.dt.strftime("%Y-%m-%d %H:%M").values
    plt.suptitle(f"Precipitation Comparison at {time_str}")
    plt.tight_layout()
    plt.show()

def create_precipitation_animation(combined_ds, num_frames=24, save_path=None):
    if len(combined_ds.time) < num_frames:
        num_frames = len(combined_ds.time)
    ds_subset = combined_ds.isel(time=slice(0, num_frames)).copy()

    ds_subset["era5_tp"] = ds_subset["era5_tp"] * 1000

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    era5_max = ds_subset["era5_tp"].max().item()
    pred_max = ds_subset["pred_precipitation"].max().item()
    global_max = max(era5_max, pred_max)

    ds_0 = ds_subset.isel(time=0)

    era5_plot = ds_0["era5_tp"].plot(
        ax=axes[0], cmap="YlGnBu", vmin=0, vmax=global_max,
        cbar_kwargs={"label": "Precipitation (mm)"}, add_colorbar=True
    )
    axes[0].set_title("Ground Truth (ERA5 Interpolated)")

    pred_plot = ds_0["pred_precipitation"].plot(
        ax=axes[1], cmap="YlGnBu", vmin=0, vmax=global_max,
        cbar_kwargs={"label": "Precipitation (mm)"}, add_colorbar=True
    )
    axes[1].set_title("Prediction (SpateGAN)")

    time_str = ds_0.time.dt.strftime("%Y-%m-%d %H:%M").item()
    title = fig.suptitle(f"Precipitation Comparison at {time_str}")

    def update(frame_index):
        ds_frame = ds_subset.isel(time=frame_index)

        era5_plot.set_array(ds_frame["era5_tp"].values.ravel())
        pred_plot.set_array(ds_frame["pred_precipitation"].values.ravel())

        new_time_str = ds_frame.time.dt.strftime("%Y-%m-%d %H:%M").item()
        title.set_text(f"Precipitation Comparison at {new_time_str}")

        return era5_plot, pred_plot, title

    # blit=False prevents the AttributeError with fig.suptitle
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)

    if save_path:
        writer = animation.FFMpegWriter(fps=5, bitrate=1800)
        ani.save(save_path, writer=writer)
        plt.close(fig)
    elif 'get_ipython' in globals() and 'IPython' in globals():
        from IPython.display import HTML
        plt.close(fig)
        return HTML(ani.to_jshtml())
    else:
        plt.tight_layout()
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

if __name__ == "__main__":
    era5_file = Path("data/era5_2026-03-01_to_2026-03-04_clean.nc")
    spategan_file = Path("output/utm/spateGAN_ERA5_latlon_52.12N_6.88E_20260301_20260303_e10.nc")

    if era5_file.exists() and spategan_file.exists():
        combined_dataset = visualize_matched_datasets(era5_file, spategan_file)
        visualize_precipitation_map(combined_dataset)

        # Call this before your visualization function
        check_data_health(combined_dataset, time_index=0)
        debug_spategan_nans(spategan_file, combined_dataset)
        create_precipitation_animation(combined_dataset, num_frames=24, save_path="output/precipitation_animation.mp4")
    else:
        print("Dataset paths do not exist. Please check the file locations.")
