"""
This program was created to test 
    1. "prepare_utils.py", 
    2. "prepare_rhorho.py", 
    3. "download_data_rhorho.py"

You may run it using the following command depeding on where you store all the data:    
$ python tests.py --source-1 "data" --source-2 "data_original"
"""

import argparse
from tests_py.test_data import test_parsed_data, show_example_records
from pathlib import Path

# Command line arguments needed for running the program independently
parser = argparse.ArgumentParser(description='Data tester')
parser.add_argument("--source-1", dest="SOURCE_1", type=Path, 
                    help="the first directory containing data to be compared", required=True)
parser.add_argument("--source-2", dest="SOURCE_2", type=Path, 
                    help="the second directory containing data to be compared", required=True)
args = parser.parse_args()

# Testing the parsed data
test_parsed_data(args)

# Showing example records for manual comparison with the original text files (*.outTUPLE_labFrame)
show_example_records(args)