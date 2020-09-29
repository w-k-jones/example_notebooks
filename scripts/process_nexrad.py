import os
import sys
import inspect
import itertools
import tarfile

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import pyart

# code from https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path?lq=1#comment15918105_6098238 to load a realitive folde from a notebook
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from utils import io, abi, glm, nexrad

date = datetime(2018,6,19,16)
abi_files = sorted(io.find_abi_files(date, satellite=16, product='MCMIP', view='C', mode=3,
                                        save_dir=goes_data_path,
                                        replicate_path=True, check_download=True,
                                        n_attempts=1, download_missing=True))

abi_files = {io.get_goes_date(i):i for i in abi_files}
abi_dates = list(abi_files.keys())

# Load a stack of goes datasets using xarray. Select a region over Northern Florida. (full file size in 1500x2500 pixels)
goes_ds = xr.open_mfdataset(abi_files.values(), concat_dim='t', combine='nested').isel({'x':slice(1300,1550), 'y':slice(650,900)})

# Now let's find the corresponding GLM files
glm_files = sorted(io.find_glm_files(date, satellite=16,
                                        save_dir=goes_data_path,
                                        replicate_path=True, check_download=True,
                                        n_attempts=1, download_missing=True))

glm_files = {io.get_goes_date(i):i for i in glm_files}
glm_dates = list(glm_files.keys())
len(glm_files)

glm_grid = xr.DataArray(np.zeros(goes_ds.CMI_C13.shape), goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)

for i, t in enumerate(glm_grid.t):
    glm_grid[i] = glm.get_glm_hist(glm_files, goes_ds, abi_dates[i], abi_dates[i]+timedelta(minutes=5))

# Pull out specific sites over Florida. There are a lot more sites covering the entire US (all site codes starting with 'K')
nexrad_sites = ['KTBW','KMLB','KAMX','KJAX','KVAX','KCLX','KTLH','KJGX','KEOX']
nexrad_files = sum([find_nexrad_files(date, site, save_dir=nexrad_data_path, download_missing=True)
                    for site in nexrad_sites], [])

stack_count, stack_mean = [np.stack(temp) for temp in zip(*[nexrad.get_site_grids(nf, goes_ds, abi_dates)
                                                            for nf in nexrad_files])]

ref_grid = xr.DataArray(np.nansum(stack_count*stack_mean, 0)/np.nansum(stack_count, 0),
                        goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)
ref_mask = xr.DataArray(np.nansum(stack_count, 0)>0, goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)