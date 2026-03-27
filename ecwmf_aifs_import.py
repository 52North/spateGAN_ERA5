import os
import xarray as xr
from datetime import datetime
from ecmwf.opendata import Client


date = datetime.now().strftime("%Y%m%d")
out_dir = "./data/aifs"
filename = os.path.join(out_dir, f"ecwmf_{date}_aifs.grib2")

client = Client(source="ecmwf", model="aifs-single")
client.retrieve(
    # date= date as up to day,
    # time= hours of the day,   # or just date with the time: date='2022-01-25 12:00:00'; strings and datetime objects
    step=list(range(0, 36, 6)),
    type="fc",
    param=["cp", "tp"],
    target=filename,
)

ds = xr.open_dataset(filename)
# rename vars for clarity
ds = ds.rename({
    'valid_time': 'time',
    'time': 'start_time',
})
# use time instead of steps as dim
ds = ds.swap_dims({'step': 'time'})
# calc lsp (lsp & cp used by predictor)
ds['lsp'] = ds['tp'] - ds['cp']
# data is 6 hours cumsum; resample to 1h - used by predictor
ds_hourly = ds.resample(time="1h").interpolate(kind='linear')
# de-accumulate vars
ds_hourly_rate = ds_hourly.diff(dim="time")
# data is in mm we use m
ds_m = ds_hourly_rate / 1000
# just use 'cp' and 'lsp'
# make sure no negative values for precip
ds_spateGAN = ds_m[['cp', 'lsp']].clip(min=0)


# save
ds_spateGAN.to_netcdf(os.path.join(out_dir, f"ecwmf_{date}_clean.nc"))

