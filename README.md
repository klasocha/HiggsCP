How to prepare data: case of rho-rho

Prepare data: step 1:
---------------------
Original data are in the files
    pythia.H.rhorho.1M.%s.%s.outTUPLE_labFrame
available from location   
http://th-www.if.uj.edu.pl/~erichter/forHiggsCP/HiggsCP_data/rhorho/

Prepare data: step 2:
---------------------
To convert into .npy format use script
https://github.com/klasocha/HiggsCP/blob/erichter-CPmix/src_py/prepare_rhorho.py
which will process files of each CPmix version and create separate .npy files
with events, with CP weights and with permution sequences. 

Prepare data: step 3:
---------------------
It is very handy then to append all weights into one file
This can be processed with script
https://github.com/klasocha/HiggsCP/blob/erichter-CPmix/src_py/download_data_rhorho.py

How to analyse data: case of rho-rho

Analyse data:
--------------
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
      
Checks to be completed:
-----------------------
Plotting: can be prepared as nootebook or .py files, but plots should be (also) available in .eps and .pdf format.
Try to assure publishable quality of the plots: marked axeses, legends, line/markers style.

Check-point 1: for few events plot calculated (from functional form) and original (input files) 
weights as a function of mixing angle. Calculated weights are in weights.npy file, 
original weights are in  rhorho_raw.w.npy file. 

Check-point 2: for few events plot weights using functional form using coefficients from popts.npy file  
and original (input files) weights as a function of mixing angle. 
Original weights are in  rhorho_raw.w.npy file.

Check-point 3: write down mathematical formulas (latex format) used in 
https://github.com/klasocha/HiggsCP/blob/erichter-CPmix/src_py/cpmix_utils.py

 calc_weights_and_arg_maxs
 
plot for the data the resolution on the position of maximum weight using functional form
and discrete weights, show for granularity of num_classes = 11, 25, 51

Check-point 4: verify code used in method 
https://github.com/klasocha/HiggsCP/blob/erichter-CPmix/src_py/tf_model.py

 calculate_classification_metrics
