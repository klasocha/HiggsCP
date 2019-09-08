import argparse

from download_data import download_data


def download_data_a1a1(source_url, output, force_download=False):
    download_data(source_url, output, "a1a1", force_download)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Download A1A1 data')
    parser.add_argument("-i", "--input", dest="IN", help="URL for data files folder")
    parser.add_argument("-o", "--output", dest="OUT", help="Target directory to where to download the data")
    parser.add_argument("--force_download", dest="FORCE_DOWNLOAD", action='store_true')
    args = parser.parse_args()

    download_data_a1a1(args.IN, args.OUT, args.FORCE_DOWNLOAD)
