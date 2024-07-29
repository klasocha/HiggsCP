from os import mkdir, path, linesep
from urllib.request import urlretrieve 
import config

def download(args):
    """ Download original data in a raw NPY format. """
    data_path = args.IN

    if not path.exists(data_path):
        mkdir(data_path)
    
    if args.EXP == "Z":
        for i in range(0, 21, 2):
            if i == 0:
                filename = f"pythia.Z_65_155.taupol.rhorho.1M.CPmix_0.outTUPLE_labFrame"
            elif i < 10:
                filename = f"pythia.Z_65_155.taupol.rhorho.1M.CPmix_0{i}.outTUPLE_labFrame"
            else:
                filename = f"pythia.Z_65_155.taupol.rhorho.1M.CPmix_{i}.outTUPLE_labFrame"
            filepath = path.join(data_path, filename)
        
            if path.exists(filepath) and not args.FORCE_DOWNLOAD:
                print(f"Original data file \"{filepath}\" already exists.\nDownloading has been cancelled.",
                    "If you want to force download, use \"--force_download\" option.\n", sep=linesep)
                continue

            print(f"Downloading {filename} and saving it in {data_path}/ ...", sep='\r')
            urlretrieve(config.DATA_URL + filename, filepath)
    else:
        if args.DATA_FORMAT == "v1":    
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
        
        # New data format (Run 2)
        elif args.DATA_FORMAT == "v2":
            for filename in config.DATA_RUN_2_FILES:
                filepath = path.join(data_path, filename)
                if path.exists(filepath) and not args.FORCE_DOWNLOAD:
                    print(f"Original data file \"{filepath}\" already exists.\nDownloading has been cancelled.",
                        "If you want to force download, use \"--force_download\" option.\n", sep=linesep)
                else:
                    print(f"Downloading {filename} and saving it in {data_path}/ ...", sep='\r')
                    urlretrieve(config.DATA_RUN_2_URL + filename, filepath)

    print()