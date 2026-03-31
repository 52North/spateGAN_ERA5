import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_datasets(era5_path: Path | str, spategan_path: Path | str):
    ds_era5 = xr.open_dataset(era5_path)
    ds_spategan = xr.open_dataset(spategan_path)

    var_era5 = 'tp' if 'tp' in ds_era5.data_vars else list(ds_era5.data_vars)[0]
    var_spategan = 'precipitation' if 'precipitation' in ds_spategan.data_vars else list(ds_spategan.data_vars)[0]

    era5_x = 'longitude' if 'longitude' in ds_era5.coords else 'lon'
    era5_y = 'latitude' if 'latitude' in ds_era5.coords else 'lat'

    # 1. Convert ERA5 longitude from 0..360 to -180..180 and sort
    ds_era5.coords[era5_x] = (ds_era5.coords[era5_x] + 180) % 360 - 180
    ds_era5 = ds_era5.sortby(era5_x)

    # 2. Extract the spatial boundaries from the SpateGAN data
    min_lon, max_lon = float(ds_spategan.lon.min()), float(ds_spategan.lon.max())
    min_lat, max_lat = float(ds_spategan.lat.min()), float(ds_spategan.lat.max())

    # 3. Crop ERA5 data to match the SpateGAN area (adding a 0.5 degree buffer for visual context)
    buffer = 0.5
    if ds_era5[era5_y].values[0] > ds_era5[era5_y].values[-1]:
        lat_slice = slice(max_lat + buffer, min_lat - buffer)
    else:
        lat_slice = slice(min_lat - buffer, max_lat + buffer)

    ds_era5_cropped = ds_era5.sel({
        era5_x: slice(min_lon - buffer, max_lon + buffer),
        era5_y: lat_slice
    })

    # Find the wettest timestep in SpateGAN
    spatial_dims = [d for d in ds_spategan[var_spategan].dims if d != 'time']
    wettest_idx = int(ds_spategan[var_spategan].sum(dim=spatial_dims).argmax())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    era_idx = wettest_idx if wettest_idx < len(ds_era5_cropped.time) else 0

    # Plot cropped ERA5
    ds_era5_cropped[var_era5].isel(time=era_idx).plot(
        ax=axes[0],
        x=era5_x,
        y=era5_y,
        cmap='Blues',
        cbar_kwargs={'label': 'Precipitation'}
    )
    axes[0].set_title(f"ERA5 Data (Cropped Region)")

    # Plot SpateGAN
    ds_spategan[var_spategan].isel(time=wettest_idx).plot(
        ax=axes[1],
        x='lon',
        y='lat',
        cmap='Blues',
        cbar_kwargs={'label': 'Precipitation'}
    )
    axes[1].set_title(f"SpateGAN Output (Time Idx: {wettest_idx})")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    era5_file = Path("data/era5_2026-03-01_to_2026-03-04_clean.nc")
    spategan_file = Path("output/utm/spateGAN_ERA5_latlon_52.12N_6.88E_20260301_20260303_e10.nc")

    if era5_file.exists() and spategan_file.exists():
        visualize_datasets(era5_file, spategan_file)
    else:
        print("Dataset paths do not exist. Please check the file locations.")
