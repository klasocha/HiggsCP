from os import mkdir, path, linesep
from urllib.request import urlretrieve 
import config

def download(args):
    """ Download original data in a raw NPY format. """
    data_path = args.IN

    if not path.exists(data_path):
        mkdir(data_path)
    
    for i in range(0, 21):
        if i < 10:
            filename = f"pythia.H.rhorho.1M.a.CPmix_0{i}.outTUPLE_labFrame"
        else:
            filename = f"pythia.H.rhorho.1M.a.CPmix_{i}.outTUPLE_labFrame"
        filepath = path.join(data_path, filename)
    
        if path.exists(filepath) and not args.FORCE_DOWNLOAD:
            print(f"Original data file \"{filepath}\" already exists.\nDownloading has been cancelled.",
                "If you want to force download, use \"--force_download\" option.\n", sep=linesep)
            continue

        print(f"Downloading {filename} and saving it in {data_path}/ ...", sep='\r')
        urlretrieve(config.DATA_URL + filename, filepath)
    print()