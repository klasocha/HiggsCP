import os

from download_data import download_files

DATA_URL = 'http://th-www.if.uj.edu.pl/~erichter/forHiggsCP/HiggsCP_data/rhorho/'


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
