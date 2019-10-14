from plot_utils import *



plot(ve_alpha1_lab[::100], va_alpha1_A[::100], [-0.0, 3, -0.0, 3], 0.01, filename="Fig_1_1", xlabel=r"$\alpha_1 \, (true)$", ylabel=r"$\alpha_1 \, (approx.)$")
plot(ve_x1_lab[::100], ve_x1_cms[::100], [0, 1.0, 0, 1.0], 0.01, filename="Fig_1_2", xlabel=r"$x_1 \, (lab)$", ylabel=r"$x_1 \, (a_1-\rho)$")


plot_2d_hist(ve_alpha1_lab, va_alpha1_A, [-0.0, 3, -0.0, 3], 0.01, filename="Fig_1_1", xlabel=r"$\alpha_1 \, (true)$", ylabel=r"$\alpha_1 \, (approx.)$")
plot_2d_hist(ve_x1_lab, ve_x1_cms, [0, 1.0, 0, 1.0], 0.01, filename="Fig_1_2", xlabel=r"$x_1 \, (lab)$", ylabel=r"$x_1 \, (a_1-\rho)$")


error_plot(ve_alpha1_lab, va_alpha1_A, [-1.1, 1.1, 0.0, 150000], 0.02, filename="Fig_1_1_hist", title=" " r"$\alpha_{1} \ Approx. \ 1$" "\n", barrier=1, relflag=True, absflag=False, Xlabel= "Relative Error", Ylabel= "Number of Events")
error_plot(ve_x1_lab, ve_x1_cms, [-0.051, 0.051, 0.0, 300000], 0.001, filename="Fig_1_2_hist", title=" " r"$x_{1} \ Approx. \ 1$" "\n", barrier=1, relflag=True, absflag=False, Xlabel= "Relative Error", Ylabel= "Number of Events")
