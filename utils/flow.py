import numpy as np
from numpy import ma
import pandas as pd
import xarray as xr
import cv2 as cv
from scipy import ndimage as ndi

class Flow:
    """
    Class to perform semi-lagrangian operations using optical flow
    """
    def __init__(self, dataset, smoothing_passes=1, flow_kwargs={}):
        get_flow(self, dataset, smoothing_passes, flow_kwargs)

    def get_flow(self, data, smoothing_passes, flow_kwargs):
    """
    Get both forwards and backwards optical flow vectors along the time
    dimension from an array with dimensions (time, y, x)
    """
        self.shape = self.shape
        self.flow_for = np.full(self.shape+(2,), np.nan, dtype=np.float32)
        self.flow_back = np.full(self.shape+(2,), np.nan, dtype=np.float32)

        b = dataset[0].compute().data
        for i in range(self.shape[0]-1):
            a, b = b, dataset[i+1].compute().data
            self.flow_for[i] = cv_flow(a, b, **flow_kwargs)
            self.flow_back[i+1] = cv_flow(b, a, **flow_kwargs)
            if smoothing_passes > 0:
                for j in range(smoothing_passes):
                    self._smooth_flow_step(i)

        self.flow_back[0] = -self.flow_for[0]
        self.flow_for[-1] = -self.flow_back[-1]

def to_8bit(array, vmin=None, vmax=None):
    """
    Converts an array to an 8-bit range between 0 and 255 with dtype uint8
    """
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)
    array_out = (array-vmin) * 255 / (vmax-vmin)
    return array_out.astype('uint8')

"""
Dicts to convert keyword inputs to opencv flags for flow keywords
"""
flow_flags = {'default':0, 'gaussian':cv.OPTFLOW_FARNEBACK_GAUSSIAN}

def cv_flow(a, b, pyr_scale=0.5, levels=5, winsize=16, iterations=3,
            poly_n=5, poly_sigma=1.1, flags='gaussian'):
    """
    Wrapper function for cv.calcOpticalFlowFarneback
    """
    assert flags in flow_flags, \
        f"{flags} not a valid input for flags keyword, input must be one of {list(flow_flags.keys())}"

    flow = cv.calcOpticalFlowFarneback(to_8bit(a), to_8bit(b), None,
                                       pyr_scale, levels, winsize, iterations,
                                       poly_n, poly_sigma, flow_flags[flags])
    return flow

"""
Dicts to convert keyword inputs to opencv flags for remap keywords
"""
border_modes = {'constant':cv.BORDER_CONSTANT,
                'nearest':cv.BORDER_REPLICATE,
                'reflect':cv.BORDER_REFLECT,
                'mirror':cv.BORDER_REFLECT_101,
                'wrap':cv.BORDER_WRAP,
                'isolated':cv.BORDER_ISOLATED,
                'transparent':cv.BORDER_TRANSPARENT}

interp_modes = {'nearest':cv.INTER_NEAREST,
                'linear':cv.INTER_LINEAR,
                'cubic':cv.INTER_CUBIC,
                'lanczos':cv.INTER_LANCZOS4}

def cv_remap(img, locs, border_mode='constant', interp_mode='linear',
             cval=np.nan, dtype=None):
    """
    Wrapper function for cv.remap
    """
    assert border_mode in border_modes, \
        f"{border_mode} not a valid input for border_mode keyword, input must be one of {list(border_modes.keys())}"
    assert interp_mode in interp_modes, \
        f"{interp_mode} not a valid input for border_mode keyword, input must be one of {list(interp_modes.keys())}"

    if not dtype:
        dtype = img.dtype
    out_img = np.full(locs.shape[:-1], cval, dtype=dtype)

    cv.remap(img, locs.astype(np.float32), None, nterp_modes[interp_mode],
             out_img, border_modes[border_mode], cval)
    return out_img
