import numpy as np
import matplotlib.pyplot as plt
import particle as part
import rhorho as rr
import math_utils as mu
from scipy import stats
import pandas as pd

def particle_picker(data : np.ndarray, particle: int) -> np.ndarray:
    """ Returns data about one particle

    Args:
        data: array holding all data
        particle: integer code for particle

    Returns:
        data_particle: array holding data about given particle
    """

    return data[data[:,4] == particle][:,:4]

def pT(particle_array: np.ndarray) -> np.ndarray:
    """ Returns sqrt(p_x^2 + p_y^2) of a particle

    Args:
        particle_array: particle 4-vectors for each event

    Returns:
        data_particle: particle pT for each event
    """

    return np.linalg.norm(particle_array[:,:2],axis = 1).reshape((particle_array.shape[0],1))

def pT_Transposed(particle_array: np.ndarray) -> np.ndarray:
    return np.linalg.norm(particle_array[:,:2],axis = 1)

def rms(X:np.ndarray):
    sq = X * X
    sq = np.mean(sq,axis = 0)
    return np.sqrt(sq)

def calculate_inv(particle_array: np.ndarray) -> np.ndarray:
    particle_array_sq = np.array(particle_array * particle_array)
    return np.sqrt(particle_array_sq[:,3] - np.sum(particle_array_sq[:,0:3],axis=1)).reshape((particle_array_sq.shape[0],1))

def describe(variable: np.ndarray):
    nobs,minmax,mean,variance = stats.describe(variable)[:4]
    description = "N = " + str(nobs) + "MinMax = " + str(minmax) + "Mean = " + str(mean) + "Variance = " + str(variance)
    return description

def prepare_to_hist(data_hist: np.ndarray) ->list:
    pi_plus = particle_picker(data_hist,211)
    pi_minus = particle_picker(data_hist,-211)
    pi_0= particle_picker(data_hist,111)
    arr = np.arange(0,pi_0.shape[0])
    pi_0_minus = pi_0[(np.remainder(arr,2) == 0),:]
    pi_0_plus = pi_0[np.remainder(arr,2) == 1,:]
    nu_plus= particle_picker(data_hist,16)
    nu_minus= particle_picker(data_hist,-16)
    rho_plus= pi_plus + pi_0_plus
    rho_minus= pi_minus + pi_0_minus
    rho_inv_minus = calculate_inv(rho_minus)
    rho_inv_plus = calculate_inv(rho_plus)
    tau_plus = rho_minus + nu_plus
    tau_minus = rho_plus + nu_minus
    rhorho = rho_plus + rho_minus
    rhorho_inv= calculate_inv(rhorho)
    tautau = tau_minus + tau_plus
    tautau_inv= calculate_inv(tautau)
    nunu = nu_minus + nu_plus
    nunu_inv= calculate_inv(nunu)
    L_plus = tau_minus[:,3] + tau_minus[:,2]
    L_minus = tau_minus[:,3] - tau_minus[:,2]
    P_plus = tau_plus[:,3] + tau_plus[:,2]
    P_minus = tau_plus[:, 3] - tau_plus[:, 2]
    ctCS = L_plus * P_minus - L_minus * P_plus
    ctCS *= np.absolute(tau_minus[:,2] + tau_plus[:,2])
    ctCS /= (np.linalg.norm(tautau_inv,axis = 1) *tautau[:,2])
    ctCS /= np.sqrt(np.linalg.norm(tautau_inv,axis = 1) * np.linalg.norm(tautau_inv,axis = 1) + pT_Transposed(tautau) * pT_Transposed(tautau))
    ctCS = ctCS.reshape((ctCS.shape[0],1))
    variables = list((pi_plus,pi_minus,pi_0_minus,pi_0_plus,nu_plus,nu_minus,rho_plus,rho_minus,
                 tau_plus,tau_minus,rhorho,nunu,tautau,rho_inv_minus,rho_inv_plus,rhorho_inv,tautau_inv, nunu_inv,ctCS))
    return variables

def describe_helper(series):
    splits = (str(series.describe()) + "\n RMS    " + str(rms(series))).split()
    keys, values = "", ""
    for i in range(0, len(splits), 2):
        keys += "{:8}\n".format(splits[i])
        values += "{:>8}\n".format(splits[i+1])
    return keys, values
#
# def draw_histogram(variable_H : np.ndarray,variable_Z : np.ndarray,weights , weights_Z, bins):
#     return plt.hist(pT(variables_H[i]), bins=bins,histtype = 'step',color = "black",density=True, weights = weights)


def fill_histograms(data_hist_H: np.ndarray,data_hist_Z: np.ndarray,weights: np.ndarray, weights_Z : np.ndarray) ->None:
    # Extracting data about each particle to fill histograms
    variables_H = prepare_to_hist(data_hist_H)
    variables_Z = prepare_to_hist(data_hist_Z)
    names = ("pi_plus","pi_minus","pi_0_minus","pi_0_plus","nu_plus","nu_minus","rho_plus","rho_minus",
                 "tau_plus","tau_minus","rhorho","nunu","tautau","rho_inv_minus",
             "rho_inv_plus","rhorho_inv","tautau_inv","nunu_inv","ctCS")
    u = 0
    ranges = {"pT" : np.linspace(0,100,50), "tautau_inv" : np.linspace(0,300,50), "rhorho_inv" : np.linspace(0,150,50),
              "rho_inv_minus" : np.linspace(0,1.4,50), "rho_inv_plus" : np.linspace(0,1.4,50),"nunu_inv" : np.linspace(0,100,50)}

    # Drawing and exporting histograms
    for i in range(len(variables_H)):
        if (u < len(names) - 6):
            plt.figure()
            if (names[u] in ("tautau","rhorho")):
                bins = np.linspace(0,50,50)
                plt.hist(pT(variables_H[i]), bins=bins,histtype = 'step',color = "black",density=True, weights = weights)
                plt.figtext(.95, .52,"Higgs stats \n" +  describe_helper(pd.Series(pT_Transposed(variables_H[i])))[0], {'multialignment': 'left'})
                plt.figtext(1.05, .52, describe_helper(pd.Series(pT_Transposed(variables_H[i])))[1], {'multialignment': 'right'})
                plt.hist(pT(variables_Z[i]), bins=bins, histtype='step', color="red", density=True, weights = weights_Z)
                plt.figtext(.95, .08,"Z stats \n" + describe_helper(pd.Series(pT_Transposed(variables_Z[i])))[0], {'multialignment': 'left'})
                plt.figtext(1.05, .08, describe_helper(pd.Series(pT_Transposed(variables_Z[i])))[1], {'multialignment': 'right'})
            else:
                plt.hist(pT(variables_H[i]), bins=ranges["pT"], histtype='step', color="black", density=True, weights = weights)
                plt.figtext(.95, .52,"Higgs stats \n" +  describe_helper(pd.Series(pT_Transposed(variables_H[i]).T))[0], {'multialignment': 'left'})
                plt.figtext(1.05, .52, describe_helper(pd.Series(pT_Transposed(variables_H[i])))[1], {'multialignment': 'right'})
                plt.hist(pT(variables_Z[i]), bins=ranges["pT"], histtype='step', color="red", density=True, weights = weights_Z)
                plt.figtext(.95, .08,"Z stats \n" + describe_helper(pd.Series(pT_Transposed(variables_Z[i])))[0], {'multialignment': 'left'})
                plt.figtext(1.05, .08, describe_helper(pd.Series(pT_Transposed(variables_Z[i])))[1], {'multialignment': 'right'})
            plt.title(label=names[u] + "_pT")
            print("Exporting " + names[u] + "_pT")
            plt.savefig(fname="figs/" + names[u] + "_pT", bbox_inches='tight')
            plt.close('all')
        for j in range(variables_H[i].shape[1]):
            plt.figure()
            if(names[u] in ranges):
                plt.hist(variables_H[i][:, j], bins= ranges[names[u]],histtype = 'step',color = "black",density=True, weights = weights)
                plt.figtext(.95, .52,"Higgs stats \n" +  describe_helper(pd.Series(variables_H[i][:,j]))[0], {'multialignment': 'left'})
                plt.figtext(1.05, .52, describe_helper(pd.Series(variables_H[i][:,j]))[1], {'multialignment': 'right'})
                plt.hist(variables_Z[i][:, j], bins= ranges[names[u]], histtype='step', color="red",
                         density=True, weights = weights_Z)
                plt.figtext(.95, .08,"Z stats \n" + describe_helper(pd.Series(variables_Z[i][:,j]))[0], {'multialignment': 'left'})
                plt.figtext(1.05, .08, describe_helper(pd.Series(variables_Z[i][:,j]))[1], {'multialignment': 'right'})
            else:
                bins = np.linspace(np.amin(variables_H[i][:, j]),np.amax(variables_H[i][:, j]),50)
                plt.hist(variables_H[i][:, j], bins=bins,histtype = 'step',color = "black",density=True, weights = weights)
                plt.figtext(.95, .52,"Higgs stats \n" +  describe_helper(pd.Series(variables_H[i][:,j]))[0], {'multialignment': 'left'})
                plt.figtext(1.05, .52, describe_helper(pd.Series(variables_H[i][:,j]))[1], {'multialignment': 'right'})
                plt.hist(variables_Z[i][:, j], bins=bins, histtype='step', color="red", density=True, weights = weights_Z)
                plt.figtext(.95, .08,"Z stats \n" + describe_helper(pd.Series(variables_Z[i][:,j]))[0], {'multialignment': 'left'})
                plt.figtext(1.05, .08, describe_helper(pd.Series(variables_Z[i][:,j]))[1], {'multialignment': 'right'})
            plt.title(label=names[u] + str(j))
            if(variables_H[i].shape[1] > 1):
                plt.yscale('log')
            print("Exporting " + names[u] + str(j))
            plt.savefig(fname = "figs/" + names[u] + str(j), bbox_inches='tight')
            plt.close('all')
        u += 1

# Reading data from .npy files
data = np.load("Data/rhorho_raw.data.npy")
data_Z = np.load("Data/Z_65_155.rhorho_raw.data.npy")
weights = np.load("Data/rhorho_raw.w_a.npy")
weights_Z = np.load("Data/Z_65_155.rhorho_raw.w_a.npy")
# Reshaping data to represent state of particle row by row
data_H_hist = data.reshape((data.shape[0]*6,5))
data_Z_hist = data_Z.reshape((data_Z.shape[0]*6,5))
data_conc_hist = np.concatenate((data_H_hist,data_Z_hist),axis = 0)
fill_histograms(data_H_hist,data_Z_hist,weights,weights_Z)