#!/home/users/wkjones/miniconda2/envs/flow_dev/bin/python3
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

import argparse
parser = argparse.ArgumentParser(description="""Regrid GLM and NEXRAD data to the GOES-16 projection""")
parser.add_argument('date', help='Date of hour to process', type=str)
parser.add_argument('-x0', help='Initial subset x location', default=0, type=int)
parser.add_argument('-x1', help='End subset x location', default=2500, type=int)
parser.add_argument('-y0', help='Initial subset y location', default=0, type=int)
parser.add_argument('-y1', help='End subset y location', default=1500, type=int)
parser.add_argument('-sd', help='Directory to save preprocess files',
                    default='./data/regrid', type=str)

args = parser.parse_args()
date = parse_date(args.date, fuzzy=True)
x0 = int(args.x0)
x1 = int(args.x1)
y0 = int(args.y0)
y1 = int(args.y1)
save_dir = args.sd
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

save_name = 'regrid_%s.nc' % (date.strftime('%Y%m%d_%H0000'))

save_path = os.path.join(save_dir, save_name)

# code from https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path?lq=1#comment15918105_6098238 to load a realitive folde from a notebook
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.dirname(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])))
cmd_folder = '/home/users/wkjones/python/jupyter_notebooks/example_notebooks'
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from utils import io, abi, glm, nexrad

goes_data_path = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16'
if not os.path.isdir(goes_data_path):
    os.makedirs(goes_data_path)

nexrad_data_path = '/gws/nopw/j04/eo_shared_data_vol2/scratch/radar/nexrad_l2'
if not os.path.isdir(nexrad_data_path):
    os.makedirs(nexrad_data_path)

# date = datetime(2018,6,19,16)
print('Loading ABI data')
abi_files = sorted(io.find_abi_files(date, satellite=16, product='MCMIP', view='C', mode=3,
                                        save_dir=goes_data_path,
                                        replicate_path=True, check_download=True,
                                        n_attempts=1, download_missing=True))

abi_files = {io.get_goes_date(i):i for i in abi_files}
abi_dates = list(abi_files.keys())

# Load a stack of goes datasets using xarray. Select a region over Northern Florida. (full file size in 1500x2500 pixels)
goes_ds = xr.open_mfdataset(abi_files.values(), concat_dim='t', combine='nested').isel({'x':slice(x0,x1), 'y':slice(y0,y1)})

# Now let's find the corresponding GLM files
print('Finding GLM data')
glm_files = sorted(io.find_glm_files(date, satellite=16,
                                        save_dir=goes_data_path,
                                        replicate_path=True, check_download=True,
                                        n_attempts=1, download_missing=True))

glm_files = {io.get_goes_date(i):i for i in glm_files}
glm_dates = list(glm_files.keys())
len(glm_files)

glm_grid = xr.DataArray(np.zeros(goes_ds.CMI_C13.shape), goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)

print('Processing GLM data')
for i, t in enumerate(glm_grid.t):
    glm_grid[i] = glm.get_glm_hist(glm_files, goes_ds, abi_dates[i], abi_dates[i]+timedelta(minutes=5))

print('Finding NEXRAD data')
nexrad_sites = nexrad.filter_nexrad_sites(goes_ds)
print("Number of sites in bound: %d" % len(nexrad_sites))

nexrad_files = sum([io.find_nexrad_files(date, site, save_dir=nexrad_data_path, download_missing=True)
                    for site in nexrad_sites], [])

print('Processing NEXRAD data')
# raw_count, stack_count, stack_mean = [np.stack(temp) for temp in zip(*[nexrad.get_site_grids(nf, goes_ds, abi_dates)
#                                                             for nf in nexrad_files])]
#
# ref_grid = xr.DataArray(np.nansum(stack_count*stack_mean, 0)/np.nansum(stack_count, 0),
#                         goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)
# ref_mask = xr.DataArray(np.nansum(raw_count, 0)>0, goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)

ref_total = np.zeros(goes_ds.shape)
ref_counts_raw = np.zeros(goes_ds.shape)
ref_counts_masked = np.zeros(goes_ds.shape)

for nf in nexrad_files:
    print(nf)
    raw_count, stack_count, stack_mean = zip(*[nexrad.get_site_grids(nf, goes_ds, abi_dates)
    wh = np.isfinite(stack_mean*stack_count)
    ref_total[wh] += stack_mean[wh]*stack_count[wh]
    ref_counds_raw += raw_count
    ref_counts_masked += stack_count

ref_grid = ref_total/ref_counds_masked
ref_mask = ref_counds_raw == 0
ref_grid[ref_mask] = np.nan

ref_grid = xr.DataArray(ref_grid, goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)
ref_mask = xr.DataArray(ref_mask, goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)

print ('Saving to %s' % (save_path))
dataset = xr.Dataset({'glm_freq':glm_grid, 'radar_ref':ref_grid, 'radar_mask':ref_mask})

dataset.to_netcdf(save_path)
print('Finished successfully')
