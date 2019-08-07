How to prepare data: case of rho-rho

Step 1
--------
Original data are in the files
    pythia.H.rhorho.1M.%s.%s.outTUPLE_labFrame
available from location   
http://th-www.if.uj.edu.pl/~erichter/forHiggsCP/HiggsCP_data/rhorho/

Step 2
-------
To convert into .npy format use script
https://github.com/klasocha/HiggsCP/blob/erichter-CPmix/src_py/prepare_rhorho.py
which will process files of each CPmix version and create separate .npy files
with events, with CP weights and with permution sequences. 

Step 3
--------
It is very handy then to append all weights into one file
This can be processed with script
https://github.com/klasocha/HiggsCP/blob/erichter-CPmix/src_py/download_data_rhorho.py

