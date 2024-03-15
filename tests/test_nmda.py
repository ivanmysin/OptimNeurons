import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import slib as lib
import net_params as pr


params_generators = pr.theta_generators # pr.theta_spatial_generators_soma  #
params_generators["class"] = getattr(lib, params_generators["class"])

generators = params_generators["class"](params_generators)



dt = 0.1
t = np.arange(0, 1000, dt)


gmax_nmda = np.ones(1, dtype=np.float64) #* 0.3
gnmda = np.ones_like(gmax_nmda)
a_nmda = np.ones_like(gmax_nmda) * 0.062
Mg0_b = np.ones_like(gmax_nmda) / 3.57
tau_rise_nmda = 2.0
tau_decay_nmda = 89.0


norm4g = tau_rise_nmda*tau_decay_nmda/(tau_decay_nmda - tau_rise_nmda)
print(norm4g)
Vpost = np.zeros(4, dtype=np.float64) + 60

g_Unmda = (gmax_nmda * gnmda).reshape(-1, 1) / (
             1.0 + np.exp(-a_nmda.reshape(-1, 1) * (Vpost.reshape(1, -1) - 65.0)) * Mg0_b.reshape(-1, 1))
#
# g_nmda_tot = np.sum(g_nmda, axis=0)
# E_nmda_tot = np.sum(g_nmda*60, axis=0) #/ g_nmda_tot
#
# print(g_nmda_tot)
# print(E_nmda_tot)



h_nmda = 0
gnmda = 0
g_hist = []

tau_d = 6.489890385
tau_r = 801.5798994
tau_f = 19.0939326
Uinc = 0.220334906
Erev = 60.0
pconn = 0.01
tau1r = tau_d / (tau_d - tau_r)


R = 0
X = 1
U = 0


for ts in t:
    firings = generators.get_firing(ts)

    SRpre = firings
    Spre_normed = SRpre * pconn

    y_ = R * np.exp(-dt / tau_d)

    x_ = 1 + (X - 1 + tau1r * U) * np.exp(-dt / tau_r) - tau1r * U

    u_ = U * np.exp(-dt / tau_f)

    released_mediator = U * x_ * Spre_normed
    U = u_ + Uinc * (1 - u_) * Spre_normed
    R = y_ + released_mediator
    X = x_ - released_mediator

    #released_mediator = firings


    h_nmda = h_nmda * np.exp(-dt / tau_rise_nmda) + released_mediator
    gnmda = gnmda * np.exp(-dt / tau_decay_nmda) + h_nmda

    g_hist.append(g_Unmda.ravel() * gnmda)

g_hist = np.asarray(g_hist)

plt.plot(t, g_hist)
plt.show()