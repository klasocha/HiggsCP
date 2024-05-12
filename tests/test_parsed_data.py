import os
from filecmp import cmp
from utilities.data_utils import read_np

def compare_two_files(args, filename):
    """ Compare two files having the same file name """
    filepath_1 = os.path.join(args.SOURCE_1, filename)
    filepath_2 = os.path.join(args.SOURCE_2, filename)
    print(f"\t{filepath_1} and {filepath_2} are identical: {cmp(filepath_1, filepath_2, shallow=False)}")


def test_parsed_data(args):
    """ Compare the data sets parsed from the original data using "prepare_rhorho.py"
    with the ones downloaded with the help of "download_data_rhorho.py" """
    
    print("\033[1mRunning the test comparing the data sets parsed from the original data", 
          "using \"prepare_rhorho.py\" with the ones downloaded with",
           "the help of \"download_data_rhorho.py\"...\033[0m")
    
    # Comparing the weights
    for i in range(0, 21, 2):
        if i < 10:
            filename = f"rhorho_raw.w_0{i}.npy"
        else:
            filename = f"rhorho_raw.w_{i}.npy"
        compare_two_files(args, filename)

    # Comparing the 4 momenta and particles ID
    compare_two_files(args, f"rhorho_raw.data.npy")
    
    # Comparing the permutations
    compare_two_files(args, f"rhorho_raw.perm.npy")
    print()
   

def show_example_records(args):
    """ Show several records from the given data set """
    filepaths = ["rhorho_raw.data.npy", "rhorho_raw.w_09.npy", "rhorho_raw.perm.npy"]
    data = read_np(os.path.join(args.SOURCE_2, filepaths[0]))
    weights = read_np(os.path.join(args.SOURCE_2, filepaths[1]))
    perm = read_np(os.path.join(args.SOURCE_2, filepaths[2]))
    
    print("\033[1mRunning the test printing some sample values for comparing them manually",
          "with the original data files which can be opened in any text editor (\"*.outTUPLE_labFrame\")...\033[0m")
    print(f"""\tFirst 3 values from {filepaths[0]}:
          
{data[0:3]}

\tFirst 3 values from {filepaths[1]}:
\t{weights[0:3]}

\tFirst 3 values from {filepaths[2]}:
\t{perm[0:3]}
""")