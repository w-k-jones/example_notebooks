import numpy as np
from pyproj import Proj

def get_abi_proj(dataset):
    return Proj(proj='geos', h=dataset.goes_imager_projection.perspective_point_height,
                lon_0=dataset.goes_imager_projection.longitude_of_projection_origin,
                lat_0=dataset.goes_imager_projection.latitude_of_projection_origin,
                sweep=dataset.goes_imager_projection.sweep_angle_axis)

def get_abi_lat_lon(dataset, dtype=float):
    p = get_abi_proj(dataset)
    xx, yy = np.meshgrid((dataset.x.data*dataset.goes_imager_projection.perspective_point_height).astype(dtype),
                         (dataset.y.data*dataset.goes_imager_projection.perspective_point_height).astype(dtype))
    lons, lats = p(xx, yy, inverse=True)
    lons[lons>=1E30] = np.nan
    lats[lats>=1E30] = np.nan
    return lats, lons

def get_abi_x_y(lat, lon, dataset):
    p = get_abi_proj(dataset)
    x, y = p(lon, lat)
    return x/dataset.goes_imager_projection.perspective_point_height, y/dataset.goes_imager_projection.perspective_point_height

def get_abi_ref(dataset, check=False, dtype=None):
    """
    Get reflectance values from level 1 ABI datasets (for channels 1-6)
    """
    ref = dataset.Rad * dataset.kappa0
    if check:
        DQF = dataset.DQF
        ref[DQF<0] = np.nan
        ref[DQF>1] = np.nan
    if dtype == None:
        return ref
    else:
        return ref.astype(dtype)

def get_abi_bt(dataset, check=False, dtype=None):
    """
    Get brightness temeprature values for level 1 ABI datasets (for channels 7-16)
    """
    bt = (dataset.planck_fk2 / (np.log((dataset.planck_fk1 / dataset.Rad) + 1)) - dataset.planck_bc1) / dataset.planck_bc2
    if check:
        DQF = dataset.DQF
        bt[DQF<0] = np.nan
        bt[DQF>1] = np.nan
    if dtype == None:
        return bt
    else:
        return bt.astype(dtype)

def get_abi_da(dataset, check=False, dtype=None):
    """
    Calibrate raw (level 1) ABI data to brightness temperature or reflectances depending on the channel
    """
    channel = dataset.band_id.data[0]
    if channel<7:
        dataarray = get_abi_ref(dataset, check, dtype)
    else:
        dataarray = get_abi_bt(dataset, check, dtype)
#   Add in attributes
    dataarray.attrs['goes_imager_projection'] = dataset.goes_imager_projection
    dataarray.attrs['band_id'] = dataset.band_id
    dataarray.attrs['band_wavelength'] = dataset.band_wavelength
    return dataarray

def _contrast_correction(color, contrast):
    """
    Modify the contrast of an R, G, or B color channel
    See: #www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    Input:
        C - contrast level
    """
    F = (259*(contrast + 255))/(255.*259-contrast)
    COLOR = F*(color-.5)+.5
    COLOR = np.minimum(COLOR, 1)
    COLOR = np.maximum(COLOR, 0)
    return COLOR

def get_abi_rgb(C01, C02, C03, gamma=0.4, contrast=100):
    R = np.maximum(C02, 0)
    R = np.minimum(R, 1)
    G = np.maximum(C03, 0)
    G = np.minimum(G, 1)
    B = np.maximum(C01, 0)
    B = np.minimum(B, 1)
    R = np.power(R, gamma)
    G = np.power(G, gamma)
    B = np.power(B, gamma)
    G_true = 0.48358168 * R + 0.45706946 * B + 0.06038137 * G
    G_true = np.maximum(G_true, 0)
    G_true = np.minimum(G_true, 1)
    RGB = _contrast_correction(np.stack([R, G_true, B], -1), contrast=contrast)
    return RGB
