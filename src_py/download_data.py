import os
import urllib

import numpy as np

from src_py.logging_utils import Logging

DATA_URL = 'http://th-www.if.uj.edu.pl/~erichter/forMichal/HiggsCP_data_CPmix/'

logger = Logging.get_logger()


def download_files(filenames, data_path, merged_name=None):
    merged_files = []
    for filename in filenames:
        dest = os.path.join(data_path, filename)
        if os.path.exists(dest):
            logger.info('File {} already exists, skipped. If you want to force download, '
                        'use --force-download option.'.format(dest))
        else:
            logger.info("Downloading {}".format(filename))
            urllib.urlretrieve(DATA_URL + filename, dest)
        merged_files.append(np.load(dest))
    if merged_name is not None:
        merged_files = np.stack(merged_files)
        np.save(os.path.join(data_path, merged_name), merged_files)


def download_data(args):
    data_path = args.IN
    if os.path.exists(data_path) is False:
        os.mkdir(data_path)
    download_weights(args)
    download_data_files(args)


def download_weights(args):
    data_path = args.IN
    CPmix_index = ['00', '02', '04', '06', '08', '10', '12', '14', '16', '18', '20']
    filenames = ["rhorho_raw.w_{}.npy".format(cpmix_index) for cpmix_index in CPmix_index]
    download_files(filenames, data_path, "rhorho_raw_w.npy")


def download_data_files(args):
    data_path = args.IN
    files = ['rhorho_raw.data.npy', 'rhorho_raw.perm.npy']
    download_files(files, data_path)
