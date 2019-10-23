import numpy as np
import os, errno
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from anal_utils import calculate_metrics_regr_c012s_from_file

filename_rhorho_Variant_All = 'npy/nn_rhorho_Variant-All_regr_c012s_Unweighted_False_NUM_CLASSES_10/monit_npy/'

x_list = list(range(3, 52, 2))

metrics_Variant_All = [calculate_metrics_regr_c012s_from_file(filename_rhorho_Variant_All, i) for i in x_list]
metrics_Variant_All = np.stack(metrics_Variant_All)

#binning for horisontal axis
x = np.array(x_list)


# Now start plotting metrics
# Make better plots here, add axes labels, add color labels, store into figures/*.eps, figures/*.pdf files
# Should we first convert to histograms (?)

#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_acc_Variant-All_regr"
# example plt.plot(x, metrics_Variant_All[:, 0],'o', label=r'$\sigma$' )
plt.plot(x, metrics_Variant_All[:, 0],'o', label=r'$|\Delta_{class}| < 1$')
plt.plot(x, metrics_Variant_All[:, 1],'x', label=r'$|\Delta_{class}| < 2$')
plt.plot(x, metrics_Variant_All[:, 2],'d', label=r'$|\Delta_{class}| < 3$')
plt.plot(x, metrics_Variant_All[:, 3],'v', label=r'$|\Delta_{class}| < 4$')
plt.ylim([0.0, 1.5])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel('Probability')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_acc_alphaCP_Variant-All_regr"

# example plt.plot(x, metrics_Variant_All[:, 0],'o', label=r'$\sigma$' )
plt.plot(x, metrics_Variant_All[:, 8],'o', label=r'$|\Delta\alpha^{CP}| < 0.25[rad]$')
plt.plot(x, metrics_Variant_All[:, 9],'x', label=r'$|\Delta\alpha^{CP}| < 0.50[rad]$')
plt.plot(x, metrics_Variant_All[:, 10],'d', label=r'$|\Delta\alpha^{CP}| < 0.75[rad]$')
plt.plot(x, metrics_Variant_All[:, 11],'v', label=r'$|\Delta\alpha^{CP}| < 1.0[rad]$')
plt.ylim([0.0, 1.5])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel('Probability')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_meanDelt_class_Variant-All_regr"

plt.errorbar(x, metrics_Variant_All[:,4], yerr=metrics_Variant_All[:,14], label=r'$<\Delta_{class}> [idx]$', linestyle = '', marker = 'o')
plt.plot([3,51],[0,0],linestyle = "--", color = "black")

plt.ylim([-0.5, 0.5])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel(r'$<\Delta>$ classes')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_meanDelt_alphaCP_Variant-All_regr"

plt.errorbar(x, metrics_Variant_All[:,7], yerr=metrics_Variant_All[:,15], label=r'$<\Delta \alpha^{CP}> [rad]$', linestyle = '', marker = 'o')
plt.plot([3,51],[0,0],linestyle = "--", color = "black")

plt.ylim([0.0, 0.5])
plt.xticks(x)
plt.legend()
#plt.ylim([-0.5, 0.5])
plt.ylim([-0.3, 0.3])
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$<\Delta \alpha^{CP}>$ [rad]')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_L1delt_w_Variant_All_regr"

plt.plot(x, metrics_Variant_All[:, 12],'o', label=r'$l_1$ with $wt^{norm}$')
plt.plot(x, metrics_Variant_All[:, 5],'d', label=r'$l_1$ with $wt$')


plt.ylim([0.0, 0.5])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$l_1$')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()
    
#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------


pathOUT = "figures/"
filename = "rhorho_L2delt_w_Variant_All_regr"

plt.plot(x, metrics_Variant_All[:, 13],'o', label=r'$l_2$ with $wt^{norm}$')
plt.plot(x, metrics_Variant_All[:, 6],'d', label=r'$l_2$ with $wt$')

plt.ylim([0.0, 0.5])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$l_2$')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()
#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------
 
