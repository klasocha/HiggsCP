import os
import urllib
from urlparse import urljoin

import numpy as np
import requests


def download_data(source_url, output, channel, force_download=False):
    if os.path.exists(output) is False:
        os.mkdir(output)
    download_weights(source_url, output, channel, force_download)
    download_data_files(source_url, output, channel, force_download)


def resource_exists(url):
    request = requests.get(url)
    return request.status_code == 200


def download_weights(source_url, output, channel, force_download=False):
    # CPmix_index = 0 (scalar), 10 (pseudoscalar), 20 (scalar)
    CPmix_index = ['00', '02', '04', '06', '08', '10', '12', '14', '16', '18', '20']
    weights = []
    output_weight_file = os.path.join(output, channel + '_raw.w.npy')
    if os.path.exists(output_weight_file) and not force_download:
        print 'Output weights file exists. Downloading data cancelled. ' \
              'If you want to force download use --force_download option'
        return
    for index in CPmix_index:
        filename = channel + '_raw.w_' + index + '.npy'
        print 'Downloading ' + filename
        filepath = os.path.join(output, filename)
        source_filepath = urljoin(source_url, filename)
        if resource_exists(source_filepath):
            urllib.urlretrieve(source_filepath, filepath)
        else:
            raise Exception("Could not retrieve file from " + source_filepath)
        weights.append(np.load(filepath))
    weights = np.stack(weights)
    np.save(output_weight_file, weights)


def download_data_files(source_url, output, channel, force_download=False):
    files = [channel + '_raw.data.npy', channel + '_raw.perm.npy']
    for filename in files:
        source_filepath = urljoin(source_url, filename)
        file_path = os.path.join(output, filename)
        if os.path.exists(file_path) and not force_download:
            print 'File ' + file_path + ' exists. Downloading data cancelled. ' \
                                        'If you want to force download use --force_download option'
        else:
            print 'Downloading ' + filename
            if resource_exists(source_filepath):
                urllib.urlretrieve(source_filepath, file_path)
            else:
                raise Exception("Could not retrieve file from " + source_filepath)

