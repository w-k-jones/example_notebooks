import numpy as np
import pyart
import tarfile

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
