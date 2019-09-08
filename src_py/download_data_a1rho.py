import argparse

from download_data import download_data

A1RHO_data_url = "http://th-www.if.uj.edu.pl/~erichter/forHiggsCP/HiggsCP_data/a1rho/"


def download_data_a1rho(source_url, output, force_download=False):
    download_data(source_url, output, "a1rho", force_download)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download A1Rho data')
    parser.add_argument("-i", "--input", dest="IN", help="URL for data files folder", default=A1RHO_data_url)
    parser.add_argument("-o", "--output", dest="OUT", help="Target directory to where to download the data")
    parser.add_argument("--force_download", dest="FORCE_DOWNLOAD", action='store_true')
    args = parser.parse_args()

    download_data_a1rho(args.IN, args.OUT, args.FORCE_DOWNLOAD)
