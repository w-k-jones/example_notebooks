#!/home/users/wkjones/miniconda2/envs/flow_dev2/bin/python3
import os
import sys
import inspect
import itertools
import warnings

import numpy as np
from numpy import ma
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from scipy import ndimage as ndi

import argparse
parser = argparse.ArgumentParser(description="""Regrid GLM and NEXRAD data to the GOES-16 projection""")
parser.add_argument('date', help='Date of hour to process', type=str)
parser.add_argument('days', help='Number of days to process', type=float)
parser.add_argument('-x0', help='Initial subset x location', default=0, type=int)
parser.add_argument('-x1', help='End subset x location', default=2500, type=int)
parser.add_argument('-y0', help='Initial subset y location', default=0, type=int)
parser.add_argument('-y1', help='End subset y location', default=1500, type=int)
parser.add_argument('-sd', help='Directory to save preprocess files',
                    default='../data/watershed', type=str)
parser.add_argument('-gd', help='GOES directory',
                    default='../data/GOES16', type=str)
parser.add_argument('--extend_path', help='Extend save directory using year/month/day subdirectories',
                    default=True, type=bool)

args = parser.parse_args()
start_date = parse_date(args.date, fuzzy=True)
end_date = start_date + timedelta(days=args.days)

x0 = int(args.x0)
x1 = int(args.x1)
y0 = int(args.y0)
y1 = int(args.y1)

save_dir = args.sd
# if args.extend_path:
    # save_dir = os.path.join(save_dir, start_date.strftime('%Y/%m/%d'))
if not os.path.isdir(save_dir):
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

save_name = 'detected_dccs_%s.nc' % (start_date.strftime('%Y%m%d_%H0000'))

save_path = os.path.join(save_dir, save_name)

# code from https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path?lq=1#comment15918105_6098238 to load a realitive folde from a notebook
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.dirname(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from utils import io, abi, glm
from utils.flow import Flow
from utils.dataset import get_datetime_from_coord, get_time_diff_from_coord
from utils.detection import detect_growth_markers, edge_watershed
from utils.analysis import filter_labels_by_length_and_mask
from utils.validation import get_min_dist_for_objects, get_marker_distance

goes_data_path = args.gd
if not os.path.isdir(goes_data_path):
    os.makedirs(goes_data_path)

print(datetime.now(),'Loading ABI data')
print('Saving data to:',goes_data_path)
dates = pd.date_range(start_date, end_date, freq='H', closed='left').to_pydatetime()
abi_files = sorted(sum([sum([io.find_abi_files(date, satellite=16, product='MCMIP',
                                               view='C', mode=mode,
                                               save_dir=goes_data_path,
                                               replicate_path=True, check_download=True,
                                               n_attempts=1, download_missing=True)
                             for mode in [3,6]], [])
                        for date in dates], []))

# Test with some multichannel data
ds_slice = {'x':slice(x0,x1), 'y':slice(y0,y1)}
# Load a stack of goes datasets using xarray. Select a region over Northern Florida. (full file size in 1500x2500 pixels)
goes_ds = xr.open_mfdataset(abi_files, concat_dim='t', combine='nested').isel(ds_slice)
goes_dates = get_datetime_from_coord(goes_ds.t)
# Check for invalid dates (which are given a date in 2000)
wh_valid_dates = [gd > datetime(2001,1,1) for gd in goes_dates]
if np.any(np.logical_not(wh_valid_dates)):
    warnings.warn("Missing timestep found, removing")
    goes_ds = goes_ds.isel({'t':wh_valid_dates})

print('%d files found'%len(abi_files))

# Extract fields and load into memory
print(datetime.now(),'Loading WVD')
wvd = goes_ds.CMI_C08 - goes_ds.CMI_C10
if hasattr(wvd, "compute"):
    wvd = wvd.compute()
print(datetime.now(),'Loading BT')
bt = goes_ds.CMI_C13
if hasattr(bt, "compute"):
    bt = bt.compute()
print(datetime.now(),'Loading SWD')
swd = goes_ds.CMI_C13 - goes_ds.CMI_C15
if hasattr(swd, "compute"):
    swd = swd.compute()

wh_all_missing = np.any([np.all(np.isnan(wvd), (1,2)),
                         np.all(np.isnan(bt), (1,2)),
                         np.all(np.isnan(swd), (1,2))],
                         0)
if np.any(wh_all_missing):
    warnings.warn("Missing data found at timesteps")
    goes_ds = goes_ds.isel({'t':np.logical_not(wh_all_missing)})

    print(datetime.now(),'Loading WVD')
    wvd = goes_ds.CMI_C08 - goes_ds.CMI_C10
    if hasattr(wvd, "compute"):
        wvd = wvd.compute()
    print(datetime.now(),'Loading BT')
    bt = goes_ds.CMI_C13
    if hasattr(bt, "compute"):
        bt = bt.compute()
    print(datetime.now(),'Loading SWD')
    swd = goes_ds.CMI_C13 - goes_ds.CMI_C15
    if hasattr(swd, "compute"):
        swd = swd.compute()

# Now we have all the valid timesteps, check for gaps in the time series
goes_timedelta = get_time_diff_from_coord(goes_ds.t)

if np.any([td>15.5 for td in goes_timedelta]):
    raise ValueError("Time gaps in abi data greater than 15 minutes, aborting")

print(datetime.now(),'Calculating flow field')
flow_kwargs = {'pyr_scale':0.5, 'levels':5, 'winsize':16, 'iterations':3,
               'poly_n':5, 'poly_sigma':1.1, 'flags':256}

flow = Flow(bt, flow_kwargs=flow_kwargs, smoothing_passes=3)

print(datetime.now(),'Detecting growth markers')
wvd_growth, growth_markers = detect_growth_markers(flow, wvd)

print('Growth above threshold: area =', np.sum(wvd_growth>=0.5))
print('Detected markers: area =', np.sum(growth_markers.data!=0))
print('Detected markers: n =', growth_markers.data.max())

print(datetime.now(), 'Detecting thick anvil region')
inner_watershed = edge_watershed(flow, wvd-swd+np.maximum(wvd_growth,0)*5, growth_markers!=0, -5, -15)
inner_labels = filter_labels_by_length_and_mask(flow.label(inner_watershed.data),
                                                growth_markers.data!=0, 3)
print('Detected thick anvils: area =', np.sum(inner_labels!=0))
print('Detected thick anvils: n =', inner_labels.max())

print(datetime.now(), 'Detecting thin anvil region')
outer_watershed = edge_watershed(flow, wvd+swd+np.maximum(wvd_growth,0)*5, inner_labels, 0, -10)
print('Detected thin anvils: area =', np.sum(outer_watershed!=0))

print(datetime.now(),'Processing GLM data')
# Get GLM data
# Process new GLM data
glm_files = sorted(sum([sorted(io.find_glm_files(date, satellite=16,
                                     save_dir=goes_data_path,
                                     replicate_path=True, check_download=True,
                                     n_attempts=1, download_missing=True))
                 for date in dates], []))
glm_files = {io.get_goes_date(i):i for i in glm_files}
print(datetime.now(),'Regridding GLM data')
glm_grid = glm.regrid_glm(glm_files, goes_ds, corrected=False)

print(datetime.now(),'Calculating marker distances')
marker_distance = get_marker_distance(growth_markers, time_range=3)
anvil_distance = get_marker_distance(inner_labels, time_range=3)
glm_distance = get_marker_distance(glm_grid, time_range=3)

wvd_labels = filter_labels_by_length_and_mask(flow.label(wvd>=-5), wvd.data>=-5, 3)
print("warm WVD regions: n =",wvd_labels.max())
wvd_distance = get_marker_distance(wvd_labels, time_range=3)

print(datetime.now(), 'Validating detection accuracy')
marker_pod_hist = np.histogram(marker_distance[glm_grid>0],
                               weights=glm_grid.data[glm_grid>0], bins=40,
                               range=[0,40])[0] / np.sum(glm_grid.data[glm_grid>0])
wvd_pod_hist = np.histogram(wvd_distance[glm_grid>0],
                            weights=glm_grid.data[glm_grid>0], bins=40,
                            range=[0,40])[0] / np.sum(glm_grid.data[glm_grid>0])
anvil_pod_hist = np.histogram(anvil_distance[glm_grid>0],
                              weights=glm_grid.data[glm_grid>0], bins=40,
                              range=[0,40])[0] / np.sum(glm_grid.data[glm_grid>0])

growth_min_distance = get_min_dist_for_objects(glm_distance, growth_markers)[0]
growth_far_hist = np.histogram(growth_min_distance, bins=40,
                               range=[0,40])[0] / growth_markers.data.max()
wvd_min_distance = get_min_dist_for_objects(glm_distance, wvd_labels)[0]
wvd_far_hist = np.histogram(wvd_min_distance, bins=40,
                            range=[0,40])[0] / wvd_labels.max()
anvil_min_distance = get_min_dist_for_objects(glm_distance, inner_labels)[0]
anvil_far_hist = np.histogram(anvil_min_distance, bins=40,
                              range=[0,40])[0] / inner_labels.max()

print('markers:')
print('n =', growth_markers.data.max())
print(np.sum(marker_pod_hist[:10]))
print(1-np.sum(growth_far_hist[:10]))

print('WVD:')
print('n =', wvd_labels.max())
print(np.sum(wvd_pod_hist[:10]))
print(1-np.sum(wvd_far_hist[:10]))

print('anvil:')
print('n =', inner_labels.max())
print(np.sum(anvil_pod_hist[:10]))
print(1-np.sum(anvil_far_hist[:10]))

print('total GLM flashes: ', np.sum(glm_grid.data))

# Get statistics about various properties of each label
from utils.analysis import apply_func_to_labels
max_marker_growth = apply_func_to_labels(growth_markers.data, wvd_growth, np.nanmax)
max_marker_wvd = apply_func_to_labels(growth_markers.data, wvd.data, np.nanmax)
min_marker_bt = apply_func_to_labels(growth_markers.data, bt.data, np.nanmin)
anvil_for_markers = apply_func_to_labels(growth_markers.data, inner_labels, np.nanmax)
growth_area = np.bincount(growth_markers.data.ravel())[1:]
growth_lengths = [fo[0].stop-fo[0].start for fo in ndi.find_objects(growth_markers.data)]

max_anvil_growth = apply_func_to_labels(inner_labels, wvd_growth, np.nanmax)
max_anvil_wvd = apply_func_to_labels(inner_labels, wvd.data, np.nanmax)
min_anvil_bt = apply_func_to_labels(inner_labels, bt.data, np.nanmin)
anvil_area = np.bincount(inner_labels.ravel())[1:]
anvil_lengths = [fo[0].stop-fo[0].start for fo in ndi.find_objects(inner_labels)]

print(datetime.now(), 'Preparing output')
new_coords = {'t':goes_ds.t, 'y':goes_ds.y, 'x':goes_ds.x,
              'y_image':goes_ds.y_image, 'x_image':goes_ds.x_image,
              'core_index':np.arange(1,growth_markers.data.max()+1,dtype=int),
              'anvil_index':np.arange(1,inner_labels.max()+1,dtype=int)}
dataset = xr.Dataset({
                      'wvd_growth':(('t','y','x'), wvd_growth),
                      'growth_markers':(('t','y','x'), growth_markers),
                      'thick_anvil':(('t','y','x'), inner_labels),
                      'thin_anvil':(('t','y','x'), outer_watershed),
                      'glm_grid':(('t','y','x'), glm_grid),
                      'growth_pod_hist':(('nbins',), marker_pod_hist),
                      'growth_far_hist':(('nbins',), growth_far_hist),
                      'wvd_pod_hist':(('nbins',), wvd_pod_hist),
                      'wvd_far_hist':(('nbins',), wvd_far_hist),
                      'anvil_pod_hist':(('nbins',), anvil_pod_hist),
                      'anvil_far_hist':(('nbins',), anvil_far_hist),
                      'growth_pod':np.sum(marker_pod_hist[:10]),
                      'growth_far':1-np.sum(growth_far_hist[:10]),
                      'n_growth':growth_markers.data.max(),
                      'wvd_pod':np.sum(wvd_pod_hist[:10]),
                      'wvd_far':1-np.sum(wvd_far_hist[:10]),
                      'n_wvd':wvd_labels.max(),
                      'anvil_pod':np.sum(anvil_pod_hist[:10]),
                      'anvil_far':1-np.sum(anvil_far_hist[:10]),
                      'n_anvil':inner_labels.max(),
                      'n_glm':np.sum(glm_grid.data),
                      'core_to_anvil_index':(('core_index',), anvil_for_markers),
                      'max_marker_growth':(('core_index',), max_marker_growth),
                      'max_marker_wvd':(('core_index',), max_marker_wvd),
                      'min_marker_bt':(('core_index',), min_marker_bt),
                      'growth_min_distance':(('core_index',), growth_min_distance),
                      'core_area':(('core_index',), growth_area),
                      'core_length':(('core_index',), growth_lengths),
                      'max_anvil_growth':(('anvil_index',), max_anvil_growth),
                      'max_anvil_wvd':(('anvil_index',), max_anvil_wvd),
                      'min_anvil_bt':(('anvil_index',), min_anvil_bt),
                      'anvil_min_distance':(('anvil_index',), anvil_min_distance),
                      'cores_per_anvil':(('anvil_index',), np.bincount(anvil_for_markers)[1:]),
                      'anvil_area':(('anvil_index',), anvil_area),
                      'anvil_length':(('anvil_index',), anvil_lengths),
                      },
                      new_coords)

print(datetime.now(), 'Saving to %s' % (save_path))
dataset.to_netcdf(save_path)
print(datetime.now(), 'Finished successfully')
