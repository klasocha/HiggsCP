# ρρ case

## Downloading the original data
```
$ python main.py --input "data" --type nn_rhorho --epochs 5 --features Variant-All --num_classes 11
```


## Preprocessing and training
```
$ python main.py --input "data" --type nn_rhorho --epochs 5 --features Variant-All --num_classes 11
```


## Drawing the diagrams
```
$ python plots.py --option PHISTAR-DISTRIBUTION --input "data" --output "plot_py/figures" 

$ python plots.py --option C012S-DISTRIBUTION --input "data" --output "plot_py/figures" 

$ python plots.py --option C012S-WEIGHT --input "data" --output "plot_py/figures"

```

## Modules Description
*   ```src_py/prepare_rhorho.py``` converts the downloaded original data by processing the files of each CPmix version and creating separate .npy files containing the events, CP weights and permutation sequences (for shuffling). 

*   ```src_py/download_data_rhorho.py``` downloads prepared data and combine all the weights into one file.

*   ```main.py``` manages configuration and activates required channel of analysis.

> Fitted C_0, C_1, C_2 coefficients of the functional form are stored in the ```c012s.npy``` file;
  
> Calculated weights (based on the functional form) for the required number of classes are
stored in the ```weights.npy``` file
      

## Roadmap

1. Write down the mathematical formulas (latex format) used in ```src_py/cpmix_utils.py```:
```python
def calc_weights_and_argmaxs(classes, c012s, data_len, num_classes):
    """ Calculate weights and argmax values from continuum distributions. """
    ...
    return weights, argmaxs, hits_argmaxs
``` 
*plot for the data the resolution on the position of maximum weight using functional form
and discrete weights, show for granularity of num_classes = 11, 25, 51*

2. Verify the code used in method ```src_py/tf_model.py```:
```python
def calculate_classification_metrics(pred_w, calc_w, args):
    ...
    return np.array([acc, mean, l1_delta_w, l2_delta_w])
```