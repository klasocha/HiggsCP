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

def calculate_inv(particle_array: np.ndarray) -> np.ndarray:
    particle_array *= particle_array
    return np.sqrt(particle_array[:,3] - np.sum(particle_array[:,0:2],axis=1)).reshape((particle_array.shape[0],1))

# Reading data from .npy files
data = np.load("Data/rhorho_raw.data.npy")
data_Z = np.load("Data/Z_65_155.rhorho_raw.data.npy")
# Reshaping data to represent state of particle row by row
data_H_hist = data.reshape((data.shape[0]*6,5))
data_Z_hist = data_Z.reshape((data_Z.shape[0]*6,5))
data_conc_hist = np.concatenate((data_H_hist,data_Z_hist),axis = 0)

def prepare_to_hist(data_hist: np.ndarray) ->list:
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
    rho_inv_minus = calculate_inv(rho_minus)
    rho_inv_plus = calculate_inv(rho_plus)
    tau_plus = rho_plus + nu_plus
    tau_minus = rho_minus + nu_minus
    rhorho = rho_plus + rho_minus
    rhorho_inv= calculate_inv(rhorho)
    tautau = tau_minus + tau_plus
    tautau_inv= calculate_inv(tautau)
    nunu = nu_minus + nu_plus
    nunu_inv= calculate_inv(nunu)
    variables = list((pi_plus,pi_minus,pi_0,nu_plus,nu_minus,rho_plus,rho_minus,
                 tau_plus,tau_minus,rhorho,nunu,tautau,rho_inv_minus,rho_inv_plus,rhorho_inv,tautau_inv, nunu_inv))
    return variables

def fill_histograms(data_hist_H: np.ndarray,data_hist_Z: np.ndarray) ->None:
    # Extracting data about each particle to fill histograms
    variables_H = prepare_to_hist(data_hist_H)
    variables_Z = prepare_to_hist(data_hist_Z)
    names = ("pi_plus","pi_minus","pi_0","nu_plus","nu_minus","rho_plus","rho_minus",
                 "tau_plus","tau_minus","rhorho","nunu","tautau","rho_inv_minus",
             "rho_inv_plus","rhorho_inv","tautau_inv","nunu_inv")
    u = 0
    ranges = {"pT" : np.linspace(0,100,50), "tautau_inv" : np.linspace(0,500e3,50), "rhorho_inv" : np.linspace(0,250e3,50),
              "rho_inv_minus" : np.linspace(0,500,50), "rho_inv_plus" : np.linspace(0,500,50),"nunu_inv" : np.linspace(0,500,50)}
    # Drawing and exporting histograms
    for i in range(len(variables_H)):
        if (u < len(names) - 5):
            plt.figure()
            if (names[u] in ("tautau","rhorho")):
                bins = np.linspace(0,1e6,50)
                plt.hist(pT(variables_H[i]), bins=bins,histtype = 'step',color = "black",density=True)
                plt.hist(pT(variables_Z[i]), bins=bins, histtype='step', color="red", density=True)
            else:
                plt.hist(pT(variables_H[i]), bins=ranges["pT"], histtype='step', color="black", density=True)
                plt.hist(pT(variables_Z[i]), bins=ranges["pT"], histtype='step', color="red", density=True)
            plt.title(label=names[u] + "_pT")
            print("Exporting " + names[u] + "_pT")
            plt.savefig(fname="figs/" + names[u] + "_pT")
        for j in range(variables_H[i].shape[1]):
            plt.figure()
            if(names[u] in ranges):
                plt.hist(variables_H[i][:, j], bins= ranges[names[u]],histtype = 'step',color = "black",density=True)
                plt.hist(variables_Z[i][:, j], bins= ranges[names[u]], histtype='step', color="red",
                         density=True)
            else:
                bins = np.linspace(np.amin(variables_H[i][:, j]),np.amax(variables_H[i][:, j]),50)
                plt.hist(variables_H[i][:, j], bins=bins,histtype = 'step',color = "black",density=True)
                plt.hist(variables_Z[i][:, j], bins=bins, histtype='step', color="red", density=True)
            plt.title(label=names[u] + str(j))
            if(variables_H[i].shape[1] > 1):
                plt.yscale('log')
            print("Exporting " + names[u] + str(j))
            plt.savefig(fname = "figs/" + names[u] + str(j))
    #        plt.show()
        u += 1

fill_histograms(data_H_hist,data_Z_hist)