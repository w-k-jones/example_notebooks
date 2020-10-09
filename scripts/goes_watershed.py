import os
import sys
import inspect
import itertools

import numpy as np
import pandas as pd
import xarray as xr
import cv2 as cv
from scipy import ndimage as ndi
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

import argparse
parser = argparse.ArgumentParser(description="""Regrid GLM and NEXRAD data to the GOES-16 projection""")
parser.add_argument('date', help='Date of hour to process', type=str)
parser.add_argument('days', help='Number of days to process', type=float)
parser.add_argument('-x0', help='Initial subset x location', default=0, type=int)
parser.add_argument('-x1', help='End subset x location', default=2500, type=int)
parser.add_argument('-y0', help='Initial subset y location', default=0, type=int)
parser.add_argument('-y1', help='End subset y location', default=1500, type=int)
parser.add_argument('-sd', help='Directory to save preprocess files',
                    default='./data/watershed', type=str)
parser.add_argument('-gd', help='GOES directory',
                    default='./data/GOES16', type=str)
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
if args.extend_path:
    save_dir = os.path.join(save_dir, start_date.strftime('%Y/%m/%d'))
if not os.path.isdir(save_dir):
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

save_name = 'watershed_%s.nc' % (start_date.strftime('%Y%m%d_%H0000'))

save_path = os.path.join(save_dir, save_name)

# code from https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path?lq=1#comment15918105_6098238 to load a realitive folde from a notebook
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.dirname(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from utils import io, abi
from utils.flow import Flow
from utils import legacy_flow as lf

goes_data_path = args.gd
if not os.path.isdir(goes_data_path):
    os.makedirs(goes_data_path)

# date = datetime(2018,6,19,16)
print('Loading ABI data')
dates = pd.date_range(start_date, end_date, freq='H', closed='left').to_pydatetime()
abi_files = sorted(sum([io.find_abi_files(date, satellite=16, product='MCMIP', view='C', mode=3,
                                          save_dir=goes_data_path,
                                          replicate_path=True, check_download=True,
                                          n_attempts=1, download_missing=True)
                        for date in dates], []))

print('%d files found'%len(abi_files))

abi_files = {io.get_goes_date(i):i for i in abi_files}
abi_dates = list(abi_files.keys())

# Get time separation of each file
dt = [(abi_dates[1]-abi_dates[0]).total_seconds()/60] \
     + [(abi_dates[i+2]-abi_dates[i]).total_seconds()/120 \
        for i in range(len(abi_files)-2)] \
     + [(abi_dates[-1]-abi_dates[-2]).total_seconds()/60]
dt = np.array(dt)

goes_ds = xr.open_mfdataset(abi_files.values(), concat_dim='t',
                            combine='nested').isel({'x':slice(x0,x1), 'y':slice(y0,y1)})

wvd = goes_ds.CMI_C08 - goes_ds.CMI_C10
bt = goes_ds.CMI_C13
swd = goes_ds.CMI_C13 - goes_ds.CMI_C15

print('Calculating flow field')
flow_kwargs = {'pyr_scale':0.5, 'levels':6, 'winsize':32, 'iterations':4,
               'poly_n':5, 'poly_sigma':1., 'flags':cv.OPTFLOW_FARNEBACK_GAUSSIAN}
flow = Flow(bt, flow_kwargs=flow_kwargs, smoothing_passes=3)

print('Calculating edges')
edges = flow.sobel(np.maximum(np.minimum(wvd,-5),-15), direction='uphill')

print('Calulcating WVD growth')
wvd_diff = flow.convolve(flow.diff(wvd)/dt[:,np.newaxis,np.newaxis], func=lambda x:np.nanmean(x,0))

print('Calculating markers')
markers = wvd_diff>=0.5

print('Calculating mask')
mask = ndi.binary_erosion((wvd<=-15).data.compute())

print('Watershedding')
l_flow = lf.Flow_Func(flow.flow_for[...,0], flow.flow_back[...,0],
                      flow.flow_for[...,1], flow.flow_back[...,1])
watershed = lf.flow_network_watershed(edges, markers, l_flow, mask=mask,
                                      structure=ndi.generate_binary_structure(3,1),
                                      debug_mode=True)


ref_grid = xr.DataArray(ref_grid, goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)
ref_mask = xr.DataArray(ref_mask, goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)

print ('Saving to %s' % (save_path))
dataset = xr.Dataset({'watershed':(('t','y','x'), watershed),
                      'wvd_diff':(('t','y','x'), wvd_diff),
                      'x_flow_for':(('t','y','x'), flow.flow_for[...,0]),
                      'x_flow_back':(('t','y','x'), flow.flow_back[...,0]),
                      'y_flow_for':(('t','y','x'), flow.flow_for[...,1]),
                      'y_flow_back':(('t','y','x'), flow.flow_back[...,1]),
                      },
                     goes_ds.CMI_C13.coords)
dataset.to_netcdf(save_path)
print('Finished successfully')
