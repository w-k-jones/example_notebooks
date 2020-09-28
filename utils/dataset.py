import numpy as np

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
