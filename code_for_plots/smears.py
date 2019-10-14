from plot_utils import *



for lambda_noise in [0.4, 0.8]:
    

    approx = []
    exact = []
    labels = []
    for sigma in [0.2, 0.4, 0.6, 0.8, 1]:
        vn_tau1_nu_phi = smear_expnorm(v_tau1_nu_phi, lambda_noise, 0, sigma)
        exact.append(v_tau1_nu_phi)
        approx.append(vn_tau1_nu_phi)
        labels.append("beta = {0}, sigma = {1}".format(lambda_noise, sigma))
    smear_plot_multi(exact, approx, [- np.pi, np.pi, 0.0, 350000], step=0.05, filename="lambda_{0}".format(lambda_noise), labels = labels, Xlabel= r"$|Delta\phi^{\nu_1}| \, (true-smeared)$", Ylabel= r"Number of Events", relflag=False, absflag=False)


    approx = []
    exact = []
    labels = []
    for sigma in [0.0001, 0.2, 0.6]:
        vn_tau1_nu_phi = smear_expnorm(v_tau1_nu_phi, lambda_noise, 0, sigma)
        exact.append(v_tau1_nu_phi)
        approx.append(vn_tau1_nu_phi)
        labels.append("beta = {0}, sigma = {1}".format(lambda_noise, sigma))
    smear_plot_multi(exact, approx, [- np.pi, np.pi, 0.0, 350000], step=0.05, filename="variant_comparison_lambda_{0}".format(lambda_noise), labels = labels, Xlabel= r"$|Delta\phi^{\nu_1}| \, (true-smeared)$", Ylabel= r"Number of Events", relflag=False, absflag=False)




for sigma in [0.2, 0.6]:
    

    approx = []
    exact = []
    labels = []
    for lambda_noise in [0.4, 0.8]:
        vn_tau1_nu_phi = smear_expnorm(v_tau1_nu_phi, lambda_noise, 0, sigma)
        exact.append(v_tau1_nu_phi)
        approx.append(vn_tau1_nu_phi)
        labels.append("beta = {0}, sigma = {1}".format(lambda_noise, sigma))
    smear_plot_multi(exact, approx, [- np.pi, np.pi, 0.0, 350000], step=0.05, filename="sigma_{0}".format(sigma), labels = labels, Xlabel= r"$|Delta\phi^{\nu_1}| \, (true-smeared)$", Ylabel= r"Number of Events", relflag=False, absflag=False)
