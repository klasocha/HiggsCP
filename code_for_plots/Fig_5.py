from plot_utils import *
lambda_noise = 0.4

v_tau1_nu_phi = np.zeros_like(v_tau1_nu_phi)
vn_tau1_nu_phi1 = smear_exp(v_tau1_nu_phi, lambda_noise) 
vn_tau1_nu_phi2 = smear_exp(v_tau1_nu_phi, 2*lambda_noise)

smear_plot2([v_tau1_nu_phi]*2, [vn_tau1_nu_phi1, vn_tau1_nu_phi2], [-np.pi, np.pi, 0.0, 500000], step=0.05, filename="Fig_5_1", label1= r"Variant 3.1.4 (b, c = 0),", label2= r" Variant 3.1.8 (b, c = 0),", Xlabel= r"$\Delta\phi_{\nu_1} \, (true-smeared)$", Ylabel= r"Number of Events", relflag=False, absflag=False)


vn_tau1_nu_phi2 = smear_polynomial(v_tau1_nu_phi, lambda_noise, 0.3, 0.8)

smear_plot2([v_tau1_nu_phi]*2, [vn_tau1_nu_phi1, vn_tau1_nu_phi2], [-np.pi, np.pi, 0.0, 500000], step=0.05, filename="Fig_5_2", label1= r"Variant 3.1.4 (b, c = 0),", label2= r"Variant 3.1.4 (b=0.3, c = 0.8),", Xlabel= r"$\Delta\phi_{\nu_1} \, (true-smeared)$", Ylabel= r"Number of Events", relflag=False, absflag=False)



vn_tau1_nu_phi2 = smear_expnorm(v_tau1_nu_phi, lambda_noise, 0, 0.4)


smear_plot2([v_tau1_nu_phi]*2, [vn_tau1_nu_phi1, vn_tau1_nu_phi2], [-np.pi, np.pi, 0.0, 500000], step=0.05, filename="Fig_5_3", label1= r"Variant 3.1.4 (b, c = 0),", label2= r"Variant 3.2.4 ($\sigma$ = 0.4),", Xlabel= r"$\Delta\phi_{\nu_1} \, (true-smeared)$", Ylabel= r"Number of Events", relflag=False, absflag=False)
