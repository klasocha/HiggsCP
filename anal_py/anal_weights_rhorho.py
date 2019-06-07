import numpy as np
from glob import glob
import os, errno

from anal_utils import evaluate_roc_auc  

directory = '../../HiggsCP_data/rhorho/'
w_a  = np.load(os.path.join(directory, 'rhorho_raw.w_a.npy'))
w_b  = np.load(os.path.join(directory, 'rhorho_raw.w_b.npy'))
w_00 = np.load(os.path.join(directory, 'rhorho_raw.w_00.npy'))
w_10 = np.load(os.path.join(directory, 'rhorho_raw.w_10.npy'))
w_20 = np.load(os.path.join(directory, 'rhorho_raw.w_20.npy'))

w_02 = np.load(os.path.join(directory, 'rhorho_raw.w_02.npy'))
w_04 = np.load(os.path.join(directory, 'rhorho_raw.w_04.npy'))
w_06 = np.load(os.path.join(directory, 'rhorho_raw.w_06.npy'))
w_08 = np.load(os.path.join(directory, 'rhorho_raw.w_08.npy'))
w_12 = np.load(os.path.join(directory, 'rhorho_raw.w_12.npy'))
w_14 = np.load(os.path.join(directory, 'rhorho_raw.w_14.npy'))
w_16 = np.load(os.path.join(directory, 'rhorho_raw.w_16.npy'))
w_18 = np.load(os.path.join(directory, 'rhorho_raw.w_18.npy'))

for i in range(0, 10):
    print w_a[i], w_00[i]
print '-------------------'

for i in range(0, 10):
    print w_b[i], w_10[i] 
print '-------------------'    

for i in range(0, 10):
    print w_a[i], w_20[i] 
print '-------------------'    



roc_auc = evaluate_roc_auc(w_a/(w_a+w_b), w_a, w_b)
print  'oracle              roc_auc =', roc_auc
print '-------------------'
roc_auc = evaluate_roc_auc(w_a, w_a, w_b)
print  'scalar/pseudoscalar roc_auc =', roc_auc
print '-------------------'
roc_auc = evaluate_roc_auc(w_00, w_a, w_b)
print  'scalar/pseudoscalar roc_auc =', roc_auc
print '-------------------'
roc_auc = evaluate_roc_auc(w_10, w_a, w_b)
print  'pseudoscalar/scalar roc_auc =', roc_auc
print '-------------------'
roc_auc = evaluate_roc_auc(w_20, w_a, w_b)
print  'scalar/pseudoscalar roc_auc =', roc_auc
print '-------------------'



roc_auc = evaluate_roc_auc(w_00/(w_00+w_00), w_00, w_00)
print  'oracle 00/00           roc_auc =', roc_auc
roc_auc = evaluate_roc_auc(w_00, w_00, w_00)
print  '       00/00           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_02), w_00, w_02)
print  'oracle 00/02           roc_auc =', roc_auc
roc_auc = evaluate_roc_auc(w_00, w_00, w_02)
print  '       00/02           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_04), w_00, w_04)
print  'oracle 00/04           roc_auc =', roc_auc
roc_auc = evaluate_roc_auc(w_00, w_00, w_04)
print  '       00/04           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_06), w_00, w_06)
print  'oracle 00/06           roc_auc =', roc_auc
roc_auc = evaluate_roc_auc(w_00, w_00, w_06)
print  '       00/06           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_08), w_00, w_08)
print  'oracle 00/08           roc_auc =', roc_auc
roc_auc = evaluate_roc_auc(w_00, w_00, w_08)
print  '       00/08           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_10), w_00, w_10)
print  'oracle 00/10           roc_auc =', roc_auc
roc_auc = evaluate_roc_auc(w_00, w_00, w_10)
print  '       00/10           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_12), w_00, w_12)
print  'oracle 00/12           roc_auc =', roc_auc
roc_auc = evaluate_roc_auc(w_00, w_00, w_12)
print  '       00/12           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_14), w_00, w_14)
print  'oracle 00/14           roc_auc =', roc_auc
roc_auc = evaluate_roc_auc(w_00, w_00, w_14)
print  '       00/14           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_16), w_00, w_16)
print  'oracle 00/16           roc_auc =', roc_auc
roc_auc = evaluate_roc_auc(w_00, w_00, w_16)
print  '       00/16           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_18), w_00, w_18)
print  'oracle 00/18           roc_auc =', roc_auc
roc_auc = evaluate_roc_auc(w_00, w_00, w_18)
print  '       00/18           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_20), w_00, w_20)
print  'oracle 00/20           roc_auc =', roc_auc
roc_auc = evaluate_roc_auc(w_00, w_00, w_20)
print  '       00/20           roc_auc =', roc_auc
print '-------------------'
