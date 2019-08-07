How to prepare data: case of rho-rho

Step 1:
--------
Original data are in the files
    pythia.H.rhorho.1M.%s.%s.outTUPLE_labFrame
available from location   
http://th-www.if.uj.edu.pl/~erichter/forHiggsCP/HiggsCP_data/rhorho/

Step 2:
-------
To convert into .npy format use script
https://github.com/klasocha/HiggsCP/blob/erichter-CPmix/src_py/prepare_rhorho.py
which will process files of each CPmix version and create separate .npy files
with events, with CP weights and with permution sequences. 

Step 3:
--------
It is very handy then to append all weights into one file
This can be processed with script
https://github.com/klasocha/HiggsCP/blob/erichter-CPmix/src_py/download_data_rhorho.py

How to analyse data: case of rho-rho

Step 1:
-----------
configure and execute 
https://github.com/klasocha/HiggsCP/blob/erichter-CPmix/main.py
tshis script is only managing configuration and activates required channel of analysis

example
python main.py -e 5 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --num_classes 10

Components:
------------
data pre-processing

https://github.com/klasocha/HiggsCP/blob/erichter-CPmix/train_rhorho.py

what is done

  --> fitted are A, B, C coefficients of the functional form and stored in the popts.npy file
  
  --> calculated are weights (based on the functional form) for required number of classes,
      stored in the weights.npy file
      
Check-point 1: for few events plot calculated (from functional form) and original (input files) 
weights as a function of mixing angle. Calculated weights are in weights.npy file, 
original weights are in  rhorho_raw.w.npy file. 

Check-point 2: for few events plot weights using functional form using coefficients from popts.npy file  
and original (input files) weights as a function of mixing angle. 
Original weights are in  rhorho_raw.w.npy file.
