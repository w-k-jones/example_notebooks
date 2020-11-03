import numpy as np
from dateutil.parser import parse as parse_date

def get_coord_bin_edges(coord):
    # Now set up the bin edges for the goes dataset coordinates. Note we multiply by height to convert into the Proj coords
    bins = np.zeros(coord.size+1)
    bins[:-1] += coord.data
    bins[1:] += coord.data
    bins[1:-1] /= 2
    return bins

def get_ds_bin_edges(ds, dims=None):
    if dims is None:
        dims = [coord for coord in ds.coords]
    elif isinstance(dims, str):
        dims = [dims]

    return [get_coord_bin_edges(ds.coords[dim]) for dim in dims]

def get_ds_shape(ds):
    shape = tuple([ds.coords[k].size for k in ds.coords if k in set(ds.coords.keys()).intersection(set(ds.dims))])
    return shape

def get_ds_core_coords(ds):
    coords = {k:ds.coords[k] for k in ds.coords if k in set(ds.coords.keys()).intersection(set(ds.dims))}
    return coords

def get_datetime_from_coord(coord):
    return [parse_date(t.item()) for t in coord.astype('datetime64[s]').astype(str)]
