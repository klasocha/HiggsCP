import numpy as np
from particle import Particle
from math_utils import * 


class RhoRhoEvent(object):
    def __init__(self, data, args, debug=True):
        # [n, pi-, pi0, an, pi+, pi0]

        p = [Particle(data[:, 5 * i:5 * i + 4]) for i in range(6)]
        cols = []

        def get_tau1(p):
            tau1_nu = p[0]
            tau1_pi = p[1:3]
            tau1_rho = tau1_pi[0] + tau1_pi[1]
            tau1 = tau1_rho+tau1_nu

            return tau1_nu, tau1_pi, tau1_rho, tau1

        def get_tau2(p):
            tau2_nu = p[3]
            tau2_pi = p[4:6]
            tau2_rho = tau2_pi[0] + tau2_pi[1]
            tau2 = tau2_rho+tau2_nu

            return tau2_nu, tau2_pi, tau2_rho, tau2

        p_tau1_nu, l_tau1_pi, p_tau1_rho, p_tau1 = get_tau1(p) # p- particle, l-list
        p_tau2_nu, l_tau2_pi, p_tau2_rho, p_tau2 = get_tau2(p)

        rho_rho = p_tau1_rho + p_tau2_rho
        nu_nu   = p_tau1_nu  + p_tau2_nu

        PHI, THETA = calc_angles(p_tau1_rho, rho_rho)

        # all particles boosted & rotated
        for i, idx in enumerate([0, 1, 2, 3, 4, 5]):
            part = p[idx]
#            part = boost_and_rotate(p[idx], PHI, THETA, rho_rho)
            if args.FEAT in ["Variant-1.0"]:
                if idx not in [0, 3]:
                    cols.append(part.vec)
                    
            if args.FEAT == "Variant-All":
                cols.append(part.vec)
                
            if args.FEAT in ["Variant-5.0","Variant-5.1"]:
                if idx not in [0, 3]:
                    cols.append(part.pt)

        # rho particles & recalculated mass 
        if args.FEAT == "Variant-5.1":
            for i, rho in enumerate([p_tau1_rho] + [p_tau2_rho]):
                cols.append(rho.pt)
                cols.append(rho.recalculated_mass)
                
            cols.append(rho_rho.pt)    
            cols.append(rho_rho.recalculated_mass)    

        if args.FEAT in ["Variant-5.0","Variant-5.1"]: 
            cols += [nu_nu.pt]

        # filter
        filt = (p_tau1_rho.pt >= 20) & (p_tau2_rho.pt >= 20)
        for part in (l_tau1_pi + l_tau2_pi):
            filt = filt & (part.pt >= 1)
        filt = filt.astype(np.float32)

        if args.FEAT in ["Variant-All","Variant-1.0", "Variant-5.0", "Variant-5.1"]:
            cols += [filt]


        for i in range(len(cols)):
            if len(cols[i].shape) == 1:
                cols[i] = cols[i].reshape([-1, 1])

        self.cols = np.concatenate(cols, 1)


