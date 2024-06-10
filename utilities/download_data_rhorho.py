import numpy as np
from os import mkdir, path, linesep
from urllib.request import urlretrieve 
import config

def download_data(args):
    """ Download all the data (weights and data files) in a raw NPY format. """
    data_path = args.IN
    if not path.exists(data_path):
        mkdir(data_path)
    download_weights(args)
    download_data_files(args)


def download_weights(args):
    """Download the weights: load the data files being retrieved via the given
    URL address one by one and then save all of them in a single file called
    rhorho_raw.w.npy"""

    # Preparing the arguments needing to identify files and directories
    data_path = args.IN
    output_weight_file = path.join(data_path, "rhorho_raw.w.npy")

    # Checking whether there is already the file in a destination directory
    if path.exists(output_weight_file) and not args.FORCE_DOWNLOAD:
        print("The output weights file already exists. Downloading has been cancelled.",
              "If you want to force download, use \"--force_download\" option.", 
              sep=linesep)
        return
    
    # Downloading files containing the weights one by one
    weights = []
    CPmix_index = ["00", # scalar
                "02", "04", "06", "08", 
                "10", # pseudoscalar
                "12", "14", "16", "18", 
                "20"] # scalar
    
    for index in CPmix_index:
        filename = 'rhorho_raw.w_' + index + '.npy'
        if args.EXP != "Z":
            print(f"Downloading {filename}")
        filepath = path.join(data_path, filename)
        if args.EXP != "Z":
            urlretrieve(config.DATA_URL + filename, filepath)
        weights.append(np.load(filepath))
   
    # Joining and then saving all the parts together in a single file
    weights = np.stack(weights)
    np.save(output_weight_file, weights)


def download_data_files(args):
    """ Download the data files """
    data_path = args.IN
    files = ['rhorho_raw.data.npy', 'rhorho_raw.perm.npy']

    for file in files:
        file_path = path.join(data_path, file)
        if path.exists(file_path) and not args.FORCE_DOWNLOAD:
            print(f"File {file_path} already exists. Downloading has been cancelled.",
                  "If you want to force download, use \"--force_download\" option.", 
                  sep=linesep)
        else:
            print(f'Downloading {file}')
            urlretrieve(config.DATA_URL + file, file_path)
