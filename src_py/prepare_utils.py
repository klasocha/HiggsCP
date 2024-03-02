""" This program implement the parsing mechanism utilised by prepare_rhorho.py.
It extract weights ("TUPLE w") and vectors (x, y, z, energy) from the original data """
import numpy as np


def find_first_line(lines, phrase):
    """ Find and return the index of the first line containing the given phrase.
    The function is utilised by read_raw_root() as the original data (txt files) does not
    start with the data itself, so it is necessary to skip some lines at the beginning. """
    for i, line in enumerate(lines):
        if phrase in line:
            return i


def read_raw_root(name, num_particles):
    """ Parse weights and data. Parameter "num_particles" acctually specifies
    the number of lines which are stored within a series of records (that is, it
    includes the weight line starting with "TUPLE" """
    with open(name) as f:
        lines = f.readlines()

    # Filtering out unnecessary lines.
    # The interesting lines start with the first "TUPLE" and end at "Analysed in total".
    lines = lines[find_first_line(lines, "TUPLE"): find_first_line(lines, "Analysed in total:")]
    
    # Ignoring the debug lines.
    lines = [line for line in lines if not line.startswith("Analysed:")]
    
    # Finding the indices of the lines starting with the examples description
    ids = [int(idx) for idx, line in enumerate(lines) if line.startswith("TUPLE")]

    # Ensuring there are `num_particles` particles for each example
    temp_list = [i for i in range(0, num_particles * len(ids), num_particles)]
    assert ids == temp_list, \
        f"Debugging (prepare_utils.py) - number of lines to be parsed ({num_particles}) does not match the expected value {int(len(lines) / len(ids))}"

    # If the numbers are not equal, check the last lines of pythia file 
    assert len(lines) == num_particles * len(ids), \
        f"Debugging (prepare_utils.py) - number of lines to be parsed ({len(lines)}) does not match the expected value {num_particles * len(ids)}"
    lines = [line.strip() for line in lines]

    # Parsing the weights
    num_examples = len(ids)
    weights = [float(lines[num_particles * i].strip().split()[1]) for i in range(num_examples)]
    weights = np.array(weights)

    # Parsing the data itself
    values = [list(map(float, " ".join(lines[num_particles * i + 1: num_particles * (i + 1)]).split()))
              for i in range(num_examples)]
    values = np.array(values)

    return values, weights