import numpy as np
import pyart
import tarfile
from scipy import stats

from .abi import get_abi_x_y
from .dataset import get_ds_bin_edges

def get_gates_from_tar(nexrad_archive):
    time_list = []
    alt_list = []
    lat_list = []
    lon_list = []
    ref_list = []
    with tarfile.open(nexrad_archive) as tar:
        # Loop iver each element and inspect to see if they are actual radar archive files (there is also metadata in the tar)
        for item in [name for name in tar.getnames() if name[-9:] == '_V06.ar2v']:
            try:
                radar = pyart.io.read_nexrad_archive(tar.extractfile(tar.getmember(item)),
                                                     include_fields=['reflectivity'],
                                                     delay_field_loading=True)
            except IOError:
                pass
            else:
                alt_list.append(radar.gate_altitude['data'])
                lat_list.append(radar.gate_latitude['data'])
                lon_list.append(radar.gate_longitude['data'])
                ref_list.append(radar.fields['reflectivity']['data'])

                start_time = parse_date(item[4:19], fuzzy=True)
                time_list.append([start_time+timedelta(seconds=t) for t in radar.time['data']])

                del radar

    times = np.concatenate(time_list, 0)
    alts = np.concatenate(alt_list, 0)
    lats = np.concatenate(lat_list, 0)
    lons = np.concatenate(lon_list, 0)
    refs = np.concatenate(ref_list, 0)

    return times, alts, lats, lons, refs

def map_nexrad_to_goes(nexrad_lat, nexrad_lon, nexrad_alt, goes_ds):
    rad_x, rad_y = get_abi_x_y(nexrad_lat, nexrad_lon, goes_ds)
    height = goes_ds.goes_imager_projection.perspective_point_height
    lat_0 = goes_ds.goes_imager_projection.latitude_of_projection_origin
    lon_0 = goes_ds.goes_imager_projection.longitude_of_projection_origin

    dlat = np.degrees(nexrad_alt*np.tan(np.radians(nexrad_lat-lat_0) + rad_y/height)/6.371E6)
    dlon = np.degrees(nexrad_alt*np.tan(np.radians(nexrad_lon-lon_0) + rad_x/height)/6.371E6)
    rad_x, rad_y = get_abi_x_y(nexrad_lat+dlat, nexrad_lon+dlon, goes_ds)

    return rad_x, rad_y

def get_nexrad_hist(nexrad_ref, nexrad_time, nexrad_alt, nexrad_lat, nexrad_lon,
                    goes_ds, start_time, end_time, min_alt=2500, max_alt=15000):

    wh_t = np.logical_and(nexrad_time>=start_time, nexrad_time<end_time)
    mask = np.logical_and(nexrad_alt[wh_t]>min_alt, nexrad_alt[wh_t]<max_alt)
    x,y = map_nexrad_to_goes(nexrad_lat[wh_t][mask], nexrad_lon[wh_t][mask],
                                    nexrad_alt[wh_t][mask], goes_ds)

    x_bins, y_bins = get_ds_bin_edges(goes_ds, ('x','y'))
    counts_hist = np.histogram2d(y, x, bins=(y_bins[::-1], x_bins))[0][::-1]
    ref_hist = stats.binned_statistic_dd((y, x), nexrad_ref[wh_t][mask],
                                         statistic='mean',
                                         bins=(y_bins[::-1], x_bins),
                                         expand_binnumbers=True)[0][::-1]

    return counts_hist, ref_hist

def get_site_grids(nexrad_file, goes_ds, goes_dates):
    radar_gates = get_gates_from_tar(nexrad_file)
    temp_stack = [get_nexrad_hist(*radar_gates_3, goes_ds, dt, dt+timedelta(minutes=5)) for dt in goes_dates]
    return [np.stack(temp) for temp in zip(*temp_stack)]
