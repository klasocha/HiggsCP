import numpy as np
import matplotlib.pyplot as plt

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


# Reading data from .npy files
data = np.load("Data/rhorho_raw.data.npy")
data_Z = np.load("Data/Z_65_155.rhorho_raw.data.npy")
# Reshaping data to represent state of particle row by row
data_H_hist = data.reshape((data.shape[0]*6,5))
data_Z_hist = data_Z.reshape((data_Z.shape[0]*6,5))
data_conc_hist = np.concatenate((data_H_hist,data_Z_hist),axis = 0)

def fill_histograms(data_hist: np.ndarray,mother_particle: str) ->None:
    # Extracting data about each particle to fill histograms
    pi_plus = particle_picker(data_hist,211)
    pi_minus = particle_picker(data_hist,-211)
    pi_0= particle_picker(data_hist,111)
    arr = np.arange(0,pi_0.shape[0])
    pi_0_minus = pi_0[(np.remainder(arr,2) == 0),:]
    pi_0_plus = pi_0[np.remainder(arr,2) == 1.,:]
    nu_plus= particle_picker(data_hist,16)
    nu_minus= particle_picker(data_hist,-16)
    rho_plus= pi_plus + pi_0_plus
    rho_minus= pi_minus + pi_0_minus
    rho_inv_minus = np.linalg.norm(rho_minus,axis = 1).reshape((rho_minus.shape[0],1))
    rho_inv_plus = np.linalg.norm(rho_plus,axis = 1).reshape((rho_plus.shape[0],1))
    tau_plus = rho_plus + nu_plus
    tau_minus = rho_minus + nu_minus
    rhorho = rho_plus + rho_minus
    rhorho_inv= np.linalg.norm(rhorho,axis = 1).reshape((rho_minus.shape[0],1))
    tautau = tau_minus + tau_plus
    tautau_inv= np.linalg.norm(tautau,axis = 1).reshape((rho_minus.shape[0],1))
    nunu = nu_minus + nu_plus
    nunu_inv= np.linalg.norm(nunu,axis = 1).reshape((rho_minus.shape[0],1))
    variables = list((pi_plus,pi_minus,pi_0,nu_plus,nu_minus,rho_plus,rho_minus,rho_inv_minus,rho_inv_plus,
                 tau_plus,tau_minus,rhorho,nunu,tautau,rhorho_inv,tautau_inv, nunu_inv))
    names = ("pi_plus","pi_minus","pi_0","nu_plus","nu_minus","rho_plus","rho_minus",
                 "tau_plus","tau_minus","rhorho","nunu","tautau","rho_inv_minus","rho_inv_plus","rhorho_inv","tautau_inv","nunu_inv")
    u = 0

    # Drawing and exporting histograms
    for i in variables:
        if (u < len(names) - 5):
            plt.figure()
            plt.hist(pT(i), bins=80)
            plt.title(label=names[u] + mother_particle +  "pT")
            plt.yscale('log')
            plt.savefig(fname="figs/" + names[u] + mother_particle + "pT")
        for j in range(i.shape[1]):
            plt.figure()
            plt.hist(i[:,j],bins = 80)
            plt.title(label = names[u] + mother_particle + str(j))
            plt.yscale('log')
            plt.savefig(fname = "figs/" + names[u] + mother_particle + str(j))
    #        plt.show()
        u += 1

fill_histograms(data_H_hist,"_H_")
fill_histograms(data_Z_hist,"_Z_")
fill_histograms(data_conc_hist,"_conc_")