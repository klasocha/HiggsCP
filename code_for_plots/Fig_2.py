from plot_utils import *


error_plot(pb_tau1_nu.e, va_tau1_nu_e_A, [-1.1, 1.1, 0.0, 150000], 0.02, filename="Fig_2_1", title=" " r"$E_{\nu_1} \, (\tau \to a_1 \nu)$" "\n", barrier=1, relflag=True, absflag=False, Xlabel= "Relative Error", Ylabel= "Number of Events")
error_plot(pb_tau1_nu.z, va_tau1_nu_long_A, [-1.1, 1.1, 0.0, 150000], 0.02, filename="Fig_2_2", title=" " r"$p_{\nu_1}^z \, (\tau \to a_1 \nu)$" "\n", barrier=1, relflag=True, absflag=False, Xlabel= "Relative Error", Ylabel= "Number of Events")
error_plot(ve_tau1_nu_trans, va_tau1_nu_trans_A, [-1.1, 1.1, 0.0, 300000], 0.02, filename="Fig_2_3", title=" " r"$p^T_{\nu_1} \, (\tau \to a_1 \nu)$" "\n", barrier=1, relflag=True, absflag=False, Xlabel= "Relative Error", Ylabel= "Number of Events")
error_plot(np.array([]), np.array([]), [-1.1, 1.1, 0.0, 300000], 0.02, filename="Fig_2_4", title=r"", barrier=1, relflag=True, absflag=False, Xlabel= "Relative Error", Ylabel= "Number of Events")
