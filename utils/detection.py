import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from .analysis import filter_labels_by_length
from .dataset import get_time_diff_from_coord

# Filtering of the growth metric occurs in three steps:
# 1. The mean of the growth is taken over 3 time periods (15 minutes)
# 2. The max value of the mean is extended to cover the adjacent time steps
# 3. An opening filter is applied, to remove any regions less than 3x3 pixels in size

def filtered_tdiff(flow, raw_diff):
    t_struct = np.zeros([3,3,3])
    t_struct[:,1,1] = 1
    s_struct = ndi.generate_binary_structure(2,1)[np.newaxis,...]

    filtered_diff = flow.convolve(raw_diff, structure=t_struct,
                                  func=lambda x:np.nanmean(x,0))
    filtered_diff = flow.convolve(filtered_diff, structure=t_struct,
                                  func=lambda x:np.nanmax(x,0))
    filtered_diff = ndi.grey_opening(filtered_diff, footprint=s_struct)

    return filtered_diff

# Get a mask which only picks up where the curvature field is positive or negative
def get_curvature_filter(wvd, sigma=2, threshold=0, direction='negative'):
    smoothed_wvd = ndi.gaussian_filter(wvd, (0,sigma,sigma))
    x_diff = np.zeros(wvd.shape)
    x_diff[:,:,1:-1] = np.diff(smoothed_wvd, n=2, axis=2)

    y_diff = np.zeros(wvd.shape)
    y_diff[:,1:-1] = np.diff(smoothed_wvd, n=2, axis=1)

    s_struct = ndi.generate_binary_structure(2,1)[np.newaxis,...]

    if direction=='negative':
        curvature_filter = ndi.binary_opening(
            ndi.binary_fill_holes(
                np.logical_and(x_diff<-threshold,y_diff<-threshold),
                structure=s_struct),
            structure=s_struct)
    elif direction=='positive':
        curvature_filter = ndi.binary_opening(
            ndi.binary_fill_holes(
                np.logical_and(x_diff>threshold,y_diff>threshold),
                structure=s_struct),
            structure=s_struct)
    return curvature_filter

# Detect regions of growth in the the wvd field
def detect_growth_markers(flow, wvd):
    wvd_diff_raw = flow.diff(wvd)/get_time_diff_from_coord(wvd.t)[:,np.newaxis,np.newaxis]

    wvd_diff_filtered = filtered_tdiff(flow, wvd_diff_raw)*get_curvature_filter(wvd)

    if isinstance(wvd, xr.DataArray):
        watershed_markers = np.logical_and(wvd_diff_filtered>=0.5, wvd.data>=-5)
    else:
        np.logical_and(wvd_diff_filtered>=0.5, wvd>=-5)

    marker_regions = flow.watershed(-wvd_diff_filtered,
                                      np.logical_and(wvd_diff_filtered>=0.5, wvd.data>=-5),
                                      mask=wvd_diff_filtered<0.25,
                                      structure=ndi.generate_binary_structure(3,1))

    s_struct = ndi.generate_binary_structure(2,1)[np.newaxis,...]

    marker_labels = flow.label(ndi.binary_opening(marker_regions, structure=s_struct))

    marker_labels = filter_labels_by_length(marker_labels, 3)

    if isinstance(wvd, xr.DataArray):
        wvd_diff_raw = xr.DataArray(wvd_diff_raw, wvd.coords, wvd.dims)
        marker_labels = xr.DataArray(marker_labels, wvd.coords, wvd.dims)
    
    return wvd_diff_raw, marker_labels
