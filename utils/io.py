import os
import subprocess
from glob import glob
from google.cloud import storage
import warnings
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import xarray as xr

def find_abi_blobs(date, satellite=16, product='Rad', view='C', mode=3, channel=1):
    storage_client = storage.Client()
    goes_bucket = storage_client.get_bucket('gcp-public-data-goes-%02d' % satellite)

    doy = (date - datetime(date.year,1,1)).days+1

    level = 'L1b' if product == 'Rad' else 'L2'

    blob_path = 'ABI-%s-%s%s/%04d/%03d/%02d/' % (level, product, view, date.year, doy, date.hour)
    if product == 'Rad' or product == 'CMI':
        blob_prefix = 'OR_ABI-%s-%s%s-M%1dC%02d_G%2d_s' % (level, product, view, mode, channel, satellite)
    else:
        blob_prefix = 'OR_ABI-%s-%s%s-M%1d_G%2d_s' % (level, product, view, mode, satellite)

    blobs = list(goes_bucket.list_blobs(prefix=blob_path+blob_prefix, delimiter='/'))

    return blobs

def download_goes_blobs(blob_list, save_dir='./', replicate_path=True, check_download=False,
                        n_attempts=0, clobber=False):
    for blob in blob_list:
        blob_path, blob_name = os.path.split(blob.name)

        if replicate_path:
            save_path = os.path.join(save_dir, blob_path)
        else:
            save_path = save_dir
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        save_file = os.path.join(save_path, blob_name)
        if clobber or not os.path.exists(save_file):
            blob.download_to_filename(save_file)

        if check_download:
            try:
                test_ds = xr.open_dataset(save_file)
                test_ds.close()
            except (IOError, OSError):
                warnings.warn('File download failed: '+save_file)
                os.remove(save_file)
                if n_attempts>0:
                    download_goes_blobs([blob], save_dir=save_dir, replicate_path=replicate_path,
                                        check_download=check_download, n_attempts=n_attempts-1,
                                        clobber=clobber)

def get_goes_date(filename):
    base_string = os.path.split(filename)[-1].split('_s')[-1]
    date = parse_date(base_string[:4]+'0101'+base_string[7:13]) + timedelta(days=int(base_string[4:7])-1)
    return date

def find_abi_files(date, satellite=16, product='Rad', view='C', mode=3, channel=1,
                    save_dir='./', replicate_path=True, check_download=False,
                    n_attempts=0, download_missing=False):
    blobs = find_abi_blobs(date, satellite=satellite, product=product, view=view, mode=mode, channel=channel)
    files = []
    for blob in blobs:
        blob_path, blob_name = os.path.split(blob.name)

        if replicate_path:
            save_path = os.path.join(save_dir, blob_path)
        else:
            save_path = save_dir
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        save_file = os.path.join(save_path, blob_name)
        if os.path.exists(save_file):
            files += [save_file]
        elif download_missing:
            download_goes_blobs([blob], save_dir=save_dir, replicate_path=replicate_path,
                                        check_download=check_download, n_attempts=n_attempts)
            if os.path.exists(save_file):
                files += [save_file]

    return files

def find_glm_blobs(date, satellite=16):
    storage_client = storage.Client()
    goes_bucket = storage_client.get_bucket('gcp-public-data-goes-%02d' % satellite)

    doy = (date - datetime(date.year,1,1)).days+1


    blob_path = 'GLM-L2-LCFA/%04d/%03d/%02d/' % (date.year, doy, date.hour)
    blob_prefix = 'OR_GLM-L2-LCFA_G%2d_s' % satellite

    blobs = list(goes_bucket.list_blobs(prefix=blob_path+blob_prefix, delimiter='/'))

    return blobs

def find_glm_files(date, satellite=16, save_dir='./', replicate_path=True, check_download=False,
                   n_attempts=0, download_missing=False):
    blobs = find_glm_blobs(date, satellite=satellite)
    files = []
    for blob in blobs:
        blob_path, blob_name = os.path.split(blob.name)

        if replicate_path:
            save_path = os.path.join(save_dir, blob_path)
        else:
            save_path = save_dir
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        save_file = os.path.join(save_path, blob_name)
        if os.path.exists(save_file):
            files += [save_file]
        elif download_missing:
            download_goes_blobs([blob], save_dir=save_dir, replicate_path=replicate_path,
                                        check_download=check_download, n_attempts=n_attempts)
            if os.path.exists(save_file):
                files += [save_file]

    return files



def test_find_glm_files():
    blobs = find_glm_blobs(datetime(2018,6,19,19))
    download_goes_blobs(blobs[:1], save_dir='./', replicate_path=False)
    files = find_glm_files(datetime(2018,6,19,19), save_dir='./', replicate_path=False,
                              download_missing=False)
    assert len(files) == 1, "Error running test of find_glm_files(), wrong number of files detected"
    for f in files:
        os.remove(f)

def test_find_glm_blobs():
    assert len(find_glm_blobs(datetime(2018,6,19,19))) == 180, "Error running test of find_glm_blobs(), wrong number of blobs detected"

def test_find_abi_files():
    test_date = datetime(2000,1,1,12)
    files = find_abi_files(test_date, view='C', channel=13, save_dir='./', replicate_path=False,
                              download_missing=True)
    assert len(files) == 1, "Error running test of find_abi_files(), wrong number of files detected"
    for f in files:
        os.remove(f)

def test_get_goes_date():
    test_date = datetime(2000,1,1,12)
    assert [get_goes_date(f.name) for f in find_abi_blobs(test_date, view='C', channel=1)][0] == test_date, "Error running test of get_goes_date(), file date does not match expected date"

def test_download_goes_blobs():
    test_date = datetime(2000,1,1,12)
    blobs = find_abi_blobs(test_date, view='C', channel=13)
    download_goes_blobs(blobs, save_dir='./', replicate_path=False, clobber=True)
    assert os.path.exists('./'+os.path.split(blobs[0].name)[-1]), "Error running test of download_goes_blobs(), file not located"
    os.remove('./'+os.path.split(blobs[0].name)[-1])

def test_find_abi_blobs():
    assert len(sum([find_abi_blobs(datetime(2000,1,1,12), view='C', channel=i+1) for i in range(16)], [])) == 15, "Error running test of find_abi_blobs(), wrong number of blobs detected"
    assert len(sum([find_abi_blobs(datetime(2000,1,1,12), view='F', channel=i+1) for i in range(16)], [])) == 1, "Error running test of find_abi_blobs(), wrong number of blobs detected"
