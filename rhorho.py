import numpy as np
from particle import Particle
from math_utils import * 


class RhoRhoEvent(object):
    def __init__(self, data, args, debug=True):
        # [n, pi-, pi0, an, pi+, pi0]

        p = [Particle(data[:, 5 * i:5 * i + 4]) for i in range(6)]
        cols = []
        cols_suppl = []
        labels = []
        labels_suppl = []

        def get_tau1(p):
            tau1_nu = p[0]
            tau1_pi = p[1:3]
            tau1_rho = tau1_pi[0] + tau1_pi[1]
            tau1     = tau1_rho+tau1_nu

            return tau1_nu, tau1_pi, tau1_rho, tau1

        def get_tau2(p):
            tau2_nu = p[3]
            tau2_pi = p[4:6]
            tau2_rho = tau2_pi[0] + tau2_pi[1]
            tau2 = tau2_rho+tau2_nu

            return tau2_nu, tau2_pi, tau2_rho, tau2

        p_tau1_nu, l_tau1_pi, p_tau1_rho,  p_tau1 = get_tau1(p) # p- particle, l-list
        p_tau2_nu, l_tau2_pi, p_tau2_rho,  p_tau2 = get_tau2(p)

        rho_rho = p_tau1_rho + p_tau2_rho

        PHI, THETA = calc_angles(p_tau1_rho, rho_rho)
        beta_noise = args.BETA

        # all particles boosted & rotated
        for i, idx in enumerate([0, 1, 2, 3, 4, 5]):
            part = boost_and_rotate(p[idx], PHI, THETA, rho_rho)
            if args.FEAT in ["Variant-1.0", "Variant-1.1", "Variant-2.0", "Variant-2.1", "Variant-2.2", "Variant-3.0", "Variant-3.1", "Variant-4.0", "Variant-4.1"]:
                if idx not in [0, 3]:
                    cols.append(part.vec)
            if args.FEAT == "Variant-All":
                cols.append(part.vec)
        
        
        if args.FEAT == "Variant-4.0":
            part   = boost_and_rotate(p_tau1, PHI, THETA, rho_rho)
            cols.append(part.vec)
            part   = boost_and_rotate(p_tau2, PHI, THETA, rho_rho)
            cols.append(part.vec)

        if args.FEAT == "Variant-4.1":
            lifetime = .08711
            scale  = smear_log(lifetime)/lifetime
            p_tau1.scale_vec(scale)
            part   = boost_and_rotate(p_tau1, PHI, THETA, rho_rho)
            cols.append(part.vec)
            scale  = smear_log(lifetime)/lifetime
            p_tau2.scale_vec(scale)
            part   = boost_and_rotate(p_tau2, PHI, THETA, rho_rho)
            cols.append(part.vec)

        # rho particles & recalculated mass 
        if args.FEAT == "Variant-1.1":
            for i, rho in enumerate([p_tau1_rho] + [p_tau2_rho]):
                rho = boost_and_rotate(rho, PHI, THETA, rho_rho)
                cols.append(rho.vec)
                cols.append(rho.recalculated_mass)

        if args.FEAT == "Variant-1.1":
            cols += [get_acoplanar_angle(p[1], p[2], p[4], p[5], rho_rho)]
            cols += [get_y(p[1], p[2], rho_rho), get_y(p[4], p[5], rho_rho)]

            #FIXME
            #sign_y1y2 =  get_y(p[1], p[2], rho_rho)* get_y(p[4], p[5], rho_rho)
            #if  sign_y1y2 > 0 :
            #   cols_suppl +=[get_acoplanar_angle(p[1], p[2], p[4], p[5], rho_rho)]
            #   cols_suppl +=[0.0]
            #if  sign_y1y2 < 0 :
            #   cols_suppl +=[0.0]
            #   cols_suppl +=[get_acoplanar_angle(p[1], p[2], p[4], p[5], rho_rho)]


        #------------------------------------------------------------

        pb_tau1_h = boost_and_rotate(p_tau1_rho, PHI, THETA, rho_rho)
        pb_tau2_h = boost_and_rotate(p_tau2_rho, PHI, THETA, rho_rho)
        pb_tau1_nu = boost_and_rotate(p_tau1_nu, PHI, THETA, rho_rho)
        pb_tau2_nu = boost_and_rotate(p_tau2_nu, PHI, THETA, rho_rho)

        #------------------------------------------------------------

        v_ETmiss_x = p_tau1_nu.x + p_tau2_nu.x
        v_ETmiss_y = p_tau1_nu.y + p_tau2_nu.y
        if args.FEAT == "Variant-2.2":
            cols += [v_ETmiss_x, v_ETmiss_y]

        #------------------------------------------------------------

        if args.METHOD == "A":
            va_alpha1, va_alpha2 = approx_alpha_A(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
        elif args.METHOD == "B":
            va_alpha1, va_alpha2 = approx_alpha_B(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
        elif args.METHOD == "C":
            va_alpha1, va_alpha2 = approx_alpha_C(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)

        #------------------------------------------------------------

        va_tau1_nu_long = va_alpha1 * pb_tau1_h.z
        va_tau2_nu_long = va_alpha2 * pb_tau2_h.z

        va_tau1_nu_E = approx_E_nu(pb_tau1_h, va_tau1_nu_long)
        va_tau2_nu_E = approx_E_nu(pb_tau2_h, va_tau2_nu_long)

        #------------------------------------------------------------

        va_tau1_nu_trans = np.sqrt(np.square(va_tau1_nu_E) - np.square(va_tau1_nu_long))
        va_tau2_nu_trans = np.sqrt(np.square(va_tau2_nu_E) - np.square(va_tau2_nu_long))
 

        v_tau1_nu_phi = np.arctan2(pb_tau1_nu.x, pb_tau1_nu.y) #boosted
        v_tau2_nu_phi = np.arctan2(pb_tau2_nu.x, pb_tau2_nu.y)
        vn_tau1_nu_phi = smear_exp(v_tau1_nu_phi, beta_noise)
        vn_tau2_nu_phi = smear_exp(v_tau2_nu_phi, beta_noise)

        tau1_sin_phi = np.sin(vn_tau1_nu_phi)
        tau1_cos_phi = np.cos(vn_tau1_nu_phi)
        tau2_sin_phi = np.sin(vn_tau2_nu_phi)
        tau2_cos_phi = np.cos(vn_tau2_nu_phi)

        #------------------------------------------------------------

        ve_x1_cms = pb_tau1_h.z / (pb_tau1_h + pb_tau1_nu).z
        ve_x2_cms = pb_tau2_h.z / (pb_tau2_h + pb_tau2_nu).z

        ve_alpha1_cms = 1/ve_x1_cms - 1
        ve_alpha2_cms = 1/ve_x2_cms - 1

        ve_tau1_nu_long = ve_alpha1_cms * pb_tau1_h.z
        ve_tau2_nu_long = ve_alpha2_cms * pb_tau2_h.z

        ve_tau1_nu_E = approx_E_nu(pb_tau1_h, ve_tau1_nu_long)
        ve_tau2_nu_E = approx_E_nu(pb_tau2_h, ve_tau2_nu_long)


        ve_tau1_nu_trans = np.sqrt(np.square(ve_tau1_nu_E) - np.square(ve_tau1_nu_long))
        ve_tau2_nu_trans = np.sqrt(np.square(ve_tau2_nu_E) - np.square(ve_tau2_nu_long))




        if args.FEAT in ["Variant-2.1", "Variant-2.2"]:
            cols += [va_tau1_nu_long, va_tau2_nu_long, va_tau1_nu_E, va_tau2_nu_E, va_tau1_nu_trans, va_tau2_nu_trans]
        elif args.FEAT in ["Variant-3.0", "Variant-3.1"]:
            cols += [va_tau1_nu_long, va_tau2_nu_long, va_tau1_nu_E, va_tau2_nu_E, va_tau1_nu_trans * tau1_sin_phi,
                     va_tau2_nu_trans * tau2_sin_phi, va_tau1_nu_trans * tau1_cos_phi, va_tau2_nu_trans * tau2_cos_phi]
        elif args.FEAT == "Variant-2.0":
            cols += [ve_tau1_nu_long, ve_tau2_nu_long, ve_tau1_nu_E, ve_tau2_nu_E, ve_tau1_nu_trans, ve_tau2_nu_trans]

        # filter
        filt = (p_tau1_rho.pt >= 20) & (p_tau2_rho.pt >= 20)
        for part in (l_tau1_pi + l_tau2_pi):
            filt = filt & (part.pt >= 1)
        filt = filt.astype(np.float32)

        if args.FEAT in ["Variant-1.0", "Variant-1.1", "Variant-All", "Variant-4.0", "Variant-4.1"]:
            cols += [filt]
            cols_suppl += [filt]



        elif args.FEAT in ["Variant-2.1", "Variant-2.2", "Variant-3.0", "Variant-3.1"]:
            isFilter = np.full(rho_rho.e.shape, True, dtype=bool)

            va_alpha1_A, va_alpha2_A = approx_alpha_A(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
            va_alpha1_B, va_alpha2_B = approx_alpha_B(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
            va_alpha1_C, va_alpha2_C = approx_alpha_C(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)

            va_tau1_nu_long_A = va_alpha1_A * pb_tau1_h.z
            va_tau1_nu_long_B = va_alpha1_B * pb_tau1_h.z 
            va_tau1_nu_long_C = va_alpha1_C * pb_tau1_h.z

            va_tau2_nu_long_A = va_alpha2_A * pb_tau2_h.z
            va_tau2_nu_long_B = va_alpha2_B * pb_tau2_h.z 
            va_tau2_nu_long_C = va_alpha2_C * pb_tau2_h.z

		
            va_tau1_nu_E_A = approx_E_nu(pb_tau1_h, va_tau1_nu_long_A)
            va_tau1_nu_E_B = approx_E_nu(pb_tau1_h, va_tau1_nu_long_B)
            va_tau1_nu_E_C = approx_E_nu(pb_tau1_h, va_tau1_nu_long_C)

            va_tau2_nu_E_A = approx_E_nu(pb_tau2_h, va_tau2_nu_long_A)
            va_tau2_nu_E_B = approx_E_nu(pb_tau2_h, va_tau2_nu_long_B)
            va_tau2_nu_E_C = approx_E_nu(pb_tau2_h, va_tau2_nu_long_C)

            va_tau1_nu_trans_A = np.sqrt(np.square(va_tau1_nu_E_A) - np.square(va_tau1_nu_long_A))
            va_tau1_nu_trans_B = np.sqrt(np.square(va_tau1_nu_E_B) - np.square(va_tau1_nu_long_B))
            va_tau1_nu_trans_C = np.sqrt(np.square(va_tau1_nu_E_C) - np.square(va_tau1_nu_long_C))

            va_tau2_nu_trans_A = np.sqrt(np.square(va_tau2_nu_E_A) - np.square(va_tau2_nu_long_A))
            va_tau2_nu_trans_B = np.sqrt(np.square(va_tau2_nu_E_B) - np.square(va_tau2_nu_long_B))
            va_tau2_nu_trans_C = np.sqrt(np.square(va_tau2_nu_E_C) - np.square(va_tau2_nu_long_C))
                  


            for alpha in [va_alpha1_A, va_alpha1_B, va_alpha1_C, va_alpha2_A, va_alpha2_B, va_alpha2_C]:
                isFilter *= (alpha > 0)
            for energy in [va_tau1_nu_E_A, va_tau1_nu_E_B, va_tau1_nu_E_C, va_tau2_nu_E_A, va_tau2_nu_E_B, va_tau2_nu_E_C]:
                isFilter *= (energy > 0)
            for trans_comp in [va_tau1_nu_trans_A, va_tau1_nu_trans_B, va_tau1_nu_trans_C, va_tau2_nu_trans_A, va_tau2_nu_trans_B, va_tau2_nu_trans_C]:
                isFilter *= np.logical_not(np.isnan(trans_comp))
            cols += [filt * isFilter]

        elif args.FEAT in ["Variant-2.0"]:
            isFilter = np.full(rho_rho.e.shape, True, dtype=bool)
            for alpha in [ve_alpha1_cms, ve_alpha2_cms]:
                isFilter *= (alpha > 0)
            for energy in [ve_tau1_nu_E, ve_tau2_nu_E]:
                isFilter *= (energy > 0)
            for trans_comp in [ve_tau1_nu_trans, ve_tau2_nu_trans]:
                isFilter *= np.logical_not(np.isnan(trans_comp))
            cols += [filt * isFilter]


        for i in range(len(cols)):
            if len(cols[i].shape) == 1:
                cols[i] = cols[i].reshape([-1, 1])

        self.cols = np.concatenate(cols, 1)


	if args.BETA > 0:
		vn_tau1_nu_phi = smear_polynomial(v_tau1_nu_phi, args.BETA, args.pol_b, args.pol_c)
		vn_tau2_nu_phi = smear_polynomial(v_tau2_nu_phi, args.BETA, args.pol_b, args.pol_c)

		tau1_sin_phi = np.sin(vn_tau1_nu_phi)
		tau1_cos_phi = np.cos(vn_tau1_nu_phi)
		tau2_sin_phi = np.sin(vn_tau2_nu_phi)
		tau2_cos_phi = np.cos(vn_tau2_nu_phi)

	self.valid_cols = [va_tau1_nu_trans * tau1_sin_phi, va_tau2_nu_trans * tau2_sin_phi, va_tau1_nu_trans * tau1_cos_phi, va_tau2_nu_trans * tau2_cos_phi]

	if args.FEAT == "Variant-1.0":
            self.labels = ["tauM_piM_px", "tauM_piM_py", "tauM_piM_pz", "tauM_piM_e", "tauM_pi0_px", "tauM_pi0_py", "tauM_pi0_pz", "tauM_pi0_e",
                           "tauP_piP_px", "tauP_piP_py", "tauP_piP_pz", "tauP_piP_e", "tauP_pi0_px", "tauP_pi0_py", "tauP_pi0_pz", "tauP_pi0_e"]
        elif args.FEAT == "Variant-1.1":
            self.labels = ["tauM_piM_px", "tauM_piM_py", "tauM_piM_pz", "tauM_piM_e", "tauM_pi0_px", "tauM_pi0_py", "tauM_pi0_pz", "tauM_pi0_e",
                           "tauP_piP_px", "tauP_piP_py", "tauP_piP_pz", "tauP_piP_e", "tauP_pi0_px", "tauP_pi0_py", "tauP_pi0_pz", "tauP_pi0_e",
                           "tauM_rho_px", "tauM_rho_py", "tauM_rho_pz", "tauM_rho_e", "tauM_rho_mass",
                           "tauP_rho_px", "tauP_rho_py", "tauP_rho_pz", "tauP_rho_e", "tauP_rho_mass",
                           "acoplanar_angle", "y1", "y2"]
            #FIXME
            #self.labels_suppl = ["acoplanar_angle_y1y2gt0","acoplanar_angle_y1y2lt0"]
            
        elif args.FEAT ==  "Variant-2.0":
            self.labels = ["tauM_piM_px", "tauM_piM_py", "tauM_piM_pz", "tauM_piM_e", "tauM_pi0_px", "tauM_pi0_py", "tauM_pi0_pz", "tauM_pi0_e",
                           "tauP_piP_px", "tauP_piP_py", "tauP_piP_pz", "tauP_piP_e", "tauP_pi0_px", "tauP_pi0_py", "tauP_pi0_pz", "tauP_pi0_e",
                           "tauM_nu_plong", "tauP_nu_plong", "tauM_nu_e", "tauP_nu_e", "tauM_nu_ptrans", "tauP_nu_ptrans"]
        elif args.FEAT ==  "Variant-2.1":
            self.labels = ["tauM_piM_px", "tauM_piM_py", "tauM_piM_pz", "tauM_piM_e", "tauM_pi0_px", "tauM_pi0_py", "tauM_pi0_pz", "tauM_pi0_e",
                           "tauP_piP_px", "tauP_piP_py", "tauP_piP_pz", "tauP_piP_e", "tauP_pi0_px", "tauP_pi0_py", "tauP_pi0_pz", "tauP_pi0_e",
                           "tauM_nu_approx_plong", "tauP_nu_approx_plong", "tauM_nu_approx_e", "tauP_nu_approx_e", "tauM_nu_approx_ptrans", "tauP_nu_approx_ptrans"]
        elif args.FEAT ==  "Variant-2.2":
            self.labels = [ "tauM_piM_px", "tauM_piM_py", "tauM_piM_pz", "tauM_piM_e", "tauM_pi0_px", "tauM_pi0_py", "tauM_pi0_pz", "tauM_pi0_e",
                            "tauP_piP_px", "tauP_piP_py", "tauP_piP_pz", "tauP_piP_e", "tauP_pi0_px", "tauP_pi0_py", "tauP_pi0_pz", "tauP_pi0_e",
                            "ETmiss_px", "ETmiss_py",
                            "tauM_nu_approx_plong", "tauP_nu_approx_plong", "tauM_nu_approx_e", "tauP_nu_approx_e", "tauM_nu_approx_ptrans", "tauP_nu_approx_ptrans"]

        elif args.FEAT ==  "Variant-3.0":
            self.labels = ["pi- x", "pi- y", "pi- z", "pi- en", "first pi0 x", "first pi0 y", "first pi0 z", "first pi0 en", "pi+ x", "pi+ y", "pi+ z", "pi+ en",
                           "second pi0 x", "second pi0 y", "second pi0 z", "second pi0 en", "va_tau1_nu_long", "va_tau2_nu_long", "va_tau1_nu_E", "va_tau2_nu_E",
                           "va_tau1_nu_trans * tau1_sin_phi", "va_tau2_nu_trans * tau2_sin_phi", "va_tau1_nu_trans * tau1_cos_phi", "va_tau2_nu_trans * tau2_cos_phi"]
        elif args.FEAT ==  "Variant-3.1":
            self.labels = ["pi- x", "pi- y", "pi- z", "pi- en", "first pi0 x", "first pi0 y", "first pi0 z", "first pi0 en", "pi+ x", "pi+ y", "pi+ z", "pi+ en",
                           "second pi0 x", "second pi0 y", "second pi0 z", "second pi0 en", "va_tau1_nu_long", "va_tau2_nu_long", "va_tau1_nu_E", "va_tau2_nu_E",
                           "va_tau1_nu_trans * tau1_sin_phi", "va_tau2_nu_trans * tau2_sin_phi", "va_tau1_nu_trans * tau1_cos_phi", "va_tau2_nu_trans * tau2_cos_phi"]

        elif args.FEAT ==  "Variant-4.0":
            self.labels = ["tauM_piM_px", "tauM_piM_py", "tauM_piM_pz", "tauM_piM_e", "tauM_pi0_px", "tauM_pi0_py", "tauM_pi0_pz", "tauM_pi0_e",
                           "tauP_piP_px", "tauP_piP_py", "tauP_piP_pz", "tauP_piP_e", "tauP_pi0_px", "tauP_pi0_py", "tauP_pi0_pz", "tauP_pi0_e",                          
                           "tauM_px", "tauM_py", "tauM_pz", "tauM_e", "tauP_px", "tauP_py", "tauP_pz", "tauP_e"]

        elif args.FEAT ==  "Variant-4.1":
            self.labels = ["tauM_piM_px", "tauM_piM_py", "tauM_piM_pz", "tauM_piM_e", "tauM_pi0_px", "tauM_pi0_py", "tauM_pi0_pz", "tauM_pi0_e",
                           "tauP_piP_px", "tauP_piP_py", "tauP_piP_pz", "tauP_piP_e", "tauP_pi0_px", "tauP_pi0_py", "tauP_pi0_pz", "tauP_pi0_e",                          
                           "tauM_approx_px", "tauM_approx_py", "tauM_approx_pz", "tauM_approx_e", "tauP_approx_px", "tauP_approx_py", "tauP_approx_pz", "tauP_approx_e"]

        elif args.FEAT == "Variant-All":
            self.labels = ["tauM_nu_px", "tauM_nu_py", "tauM_nu_pz", "tauM_nu_e",
                           "tauM_piM_px", "tauM_piM_py", "tauM_piM_pz", "tauM_piM_e", "tauM_pi0_px", "tauM_pi0_py", "tauM_pi0_pz", "tauM_pi0_e",
                           "tauP_nu_px", "tauP_nu_py", "tauP_nu_pz", "tauP_nu_e", 
                           "tauP_piP_px", "tauP_piP_py", "tauP_piP_pz", "tauP_piP_e", "tauP_pi0_px", "tauP_pi0_py", "tauP_pi0_pz", "tauP_pi0_e"]                         

