import numpy as np
import matplotlib.pyplot as plt
import slib as lib
import net_params as pr
import time
from copy import deepcopy
from scipy.optimize import differential_evolution
import h5py

COUNTER = 0


def log_cosh(y_true, y_pred):
    x = y_pred - y_true
    y = np.mean(x + np.log(1 + np.exp(-2.0 * x)) - np.log(2.0))

    return y
def get_target_Esyn(t, tc, dt, theta_freq, v_an, params):
    ALPHA = 5.0
    # meanSR = params['mean_firing_rate']
    phase = np.deg2rad(params['phase_out_place'])
    kappa = 0.15  # self.r2kappa(params["R_place_cell"])
    maxFiring = 20  # params['peak_firing_rate']

    SLOPE = np.deg2rad(params['precession_slope'] * v_an * 0.001)  # rad / ms
    ONSET = np.deg2rad(params['precession_onset'])

    sigma_spt = params['sigma_place_field'] / v_an * 1000

    mult4time = 2 * np.pi * theta_freq * 0.001

    # I0 = bessel_i0(kappa)
    normalizator = 5  # meanSR / I0 * 0.001 * dt

    amp = 20  # 2 * (maxFiring - meanSR) / (meanSR + 1)  # maxFiring / meanSR - 1 #  range [-1, inf]

    # print(meanSR)
    multip = amp * np.exp(-0.5 * ((t - tc) / sigma_spt) ** 2)

    start_place = t - tc - 3 * sigma_spt
    end_place = t - tc + 3 * sigma_spt
    inplace = 0.25 * (1.0 - (start_place / (ALPHA + np.abs(start_place)))) * (
            1.0 + end_place / (ALPHA + np.abs(end_place)))

    precession = SLOPE * t * inplace
    phases = phase * (1 - inplace) - ONSET * inplace

    Esyn = multip + normalizator * np.cos(mult4time * t + precession - phases)

    return Esyn
def simulate(X, Duration, dt, Cm, animal_velocity, params_generators, params_synapses):
    params_generators = deepcopy(params_generators)
    params_synapses = deepcopy(params_synapses)
    x_idx = 0
    for gen in params_generators["params"][1:]:
        gen["maxFiring"] = X[x_idx]
        x_idx += 1

        gen["sp_centers"] = X[x_idx] + 0.5 * 0.001 * Duration * animal_velocity
        x_idx += 1

        gen["sigma_sp"] = X[x_idx]
        x_idx += 1


    Erev = params_synapses["Erev"]
    Gmax = X[x_idx:] #params_synapses["Gmax"]

    tau_d = params_synapses["tau_d"]
    tau_r = params_synapses["tau_r"]
    tau_f = params_synapses["tau_f"]
    Uinc = params_synapses["Uinc"]
    pconn = params_synapses["pconn"]

    generators = getattr(lib, params_generators["class"])(params_generators)


    t = np.arange(0, Duration, dt)



    # gmax_nmda = np.ones(10, dtype=np.float64) #* 0.3
    # gnmda = np.ones_like(gmax_nmda)
    # a_nmda = np.ones_like(gmax_nmda) * 0.062
    # Mg0_b = np.ones_like(gmax_nmda) / 3.57
    # tau_rise_nmda = 2.0
    # tau_decay_nmda = 89.0


    #norm4g = tau_rise_nmda*tau_decay_nmda/(tau_decay_nmda - tau_rise_nmda)
    #print(norm4g)
    #Vpost = np.zeros(1, dtype=np.float64) + 60

    # g_Unmda = (gmax_nmda * gnmda).reshape(-1, 1) / (
    #              1.0 + np.exp(-a_nmda.reshape(-1, 1) * (Vpost.reshape(1, -1) - 65.0)) * Mg0_b.reshape(-1, 1))
    # h_nmda = 0
    # gnmda = 0


    tau1r = tau_d / (tau_d - tau_r)


    R = np.zeros_like(Gmax)
    X = np.ones_like(Gmax)
    U = np.zeros_like(Gmax)

    exp_tau_r = np.exp(-dt / tau_r)
    exp_tau_d = np.exp(-dt / tau_d)
    exp_tau_f = np.exp(-dt / tau_f)

    # Erev_hist = np.zeros_like(t)
    # tau_m_hist = np.zeros_like(t)
    g_hist = []


    for t_idx, ts in enumerate(t):
        SRpre = generators.get_firing(ts)

        Spre_normed = SRpre * pconn

        y_ = R * exp_tau_d

        x_ = 1 + (X - 1 + tau1r * U) * exp_tau_r - tau1r * U

        u_ = U * exp_tau_f

        released_mediator = U * x_ * Spre_normed
        U = u_ + Uinc * (1 - u_) * Spre_normed
        R = y_ + released_mediator
        X = x_ - released_mediator

        #released_mediator = firings


        # h_nmda = h_nmda * np.exp(-dt / tau_rise_nmda) + released_mediator
        # gnmda = gnmda * np.exp(-dt / tau_decay_nmda) + h_nmda
        #
        # g_hist.append(g_Unmda.ravel() * gnmda)

        g_hist.append(R)
        # g = Gmax * R
        # G_tot = np.sum(g)
        # Erevsum = np.sum(g * Erev) / (G_tot + 0.0000001)
        #
        # tau_m = G_tot / Cm
        #
        # Erev_hist[t_idx] = Erevsum
        # tau_m_hist[t_idx] = tau_m
    Erev = Erev.reshape(1, -1)

    g_hist = np.stack(g_hist)
    g_hist = Gmax * g_hist #/ np.mean(g_hist, axis=0)

    # for i in range(g_hist.shape[1]):
    #     plt.plot(t, g_hist[:, i])
    #     plt.show()


    G_tot = np.sum(g_hist, axis=1)


    Erev_hist = np.sum(g_hist*Erev, axis=1) / (G_tot + 0.1)

    tau_m_hist = Cm / (G_tot + 0.1)

    return Erev_hist, tau_m_hist
###############################################################
def Loss(X,Duration, dt, Cm, animal_velocity, params_generators, params_synapses, target_params):
    global COUNTER

    theta_freq = params_generators['params'][0]['freq']
    Erev_hist, tau_m_hist = simulate(X, Duration, dt, Cm, animal_velocity, params_generators, params_synapses)

    tc = 0.5*Duration
    t = np.linspace(0, Duration, Erev_hist.size)
    Etar = get_target_Esyn(t, tc, dt, theta_freq, animal_velocity, target_params)

    Etar = Etar[100:]
    Erev_hist = Erev_hist[100:]
    tau_m_hist = tau_m_hist[100:]


    L = 0.0
    #L += np.sqrt( np.mean( (Etar - Erev_hist)**2) )
    L += np.square(np.subtract(Etar,Erev_hist)).mean()

    #L += np.mean(np.log((Etar + 17) / (Erev_hist + 17)) ** 2)

    #L += log_cosh(Etar, Erev_hist)


    COUNTER += 1
    # if COUNTER % 32 == 0:
    #     print("COUNTER = ", COUNTER)

    return L

###############################################################
def callback(intermediate_result=None):
    #print("COUNTER = ", COUNTER)
    with h5py.File(FILE4SAVING, "w") as output:
        output.create_dataset("loss", data=intermediate_result.fun)
        output.create_dataset("X", data=intermediate_result.x)

    return False


################################################################
def get_default_x0(params):
    # initial changable params
    X0 = np.zeros(42, dtype=np.float64)
    bounds = []  # Boundaries for X
    x_names = []

    x0_idx = 0
    for neurons_types in params["neurons"]["params"][1:]:
        try:
             X0[x0_idx] = neurons_types["maxFiring"]
             bounds.append([0.0001, 100])
             x_names.append("maxFiring of {name}".format(name=neurons_types['name']) )
             x0_idx += 1

             X0[x0_idx] = neurons_types["sp_centers"]
             bounds.append([-8.0, 8.0])
             x_names.append("sp_centers of {name}".format(name=neurons_types['name']))
             x0_idx += 1

             X0[x0_idx] = neurons_types["sigma_sp"]
             bounds.append([0.1, 15])
             x_names.append("sigma_sp of {name}".format(name=neurons_types['name']))
             x0_idx += 1
        except KeyError:
            continue

    #print()
    s_size = len(params["synapses"]["Gmax"])
    X0[x0_idx: x0_idx+len(params["synapses"]["Gmax"])] = params["synapses"]["Gmax"]
    x0_idx += s_size

    #print(s_size)
    for s_idx in range(s_size):
        bounds.append([0.0, 500])
        x_names.append("Gmax {name}".format(name=params["neurons"]["params"][s_idx]['name']  ) )

    X0 = X0[:x0_idx]

    return X0, bounds, x_names
################################################################################
USE_X0_FROM_FILE = True
FILE4SAVING = "mse_results.h5"

animal_velocity = pr.V_AN
theta_freq = pr.theta_generators["params"][0]["freq"]
theta_generator = deepcopy(pr.theta_generators)
theta_generator["params"][0]["sigma_sp"] = 10000000
theta_generator["params"][0]["v_an"] = pr.V_AN
theta_generator["params"][0]["maxFiring"] = 24.0
theta_generator["params"][0]["sp_centers"] = 100000000

params_generators = {
    "class": "VonMissesSpatialMolulation",
    "name": "theta_spatial_inputs",
    "params" : [theta_generator["params"][0], ],
}


params_generators["params"].extend( deepcopy(pr.theta_spatial_generators_soma["params"]) )
params_generators["params"].extend( deepcopy(pr.theta_spatial_generators_dend["params"]) )

params_synapses = {}

Erev = []
Gmax = []
tau_d = [] #6.489890385
tau_r = [] # 801.5798994
tau_f = [] # 19.0939326
Uinc = []  #0.220334906
pconn = []  #0.220334906

for Syn in pr.synapses_params:
    for p in Syn['params']:
        Erev.append(p["Erev"])
        Gmax.append(p["gmax"])
        tau_d.append(p["tau_d"])
        tau_r.append(p["tau_r"])
        tau_f.append(p["tau_f"])
        Uinc.append(p["Uinc"])
        pconn.append(p["pconn"])

params_synapses['Erev'] = np.asarray(Erev)
params_synapses['Gmax'] = 1.0 + np.zeros_like(Erev) # np.asarray(Gmax)
params_synapses['tau_d'] = np.asarray(tau_d)
params_synapses['tau_r'] = np.asarray(tau_r)
params_synapses['tau_f'] = np.asarray(tau_f)
params_synapses['Uinc'] = np.asarray(Uinc)
params_synapses['pconn'] = np.asarray(pconn)

target_params = pr.default_param4optimization

Duration = 10000
dt = 0.1
Cm = 3.0

if USE_X0_FROM_FILE:
    _, bounds, x_names = get_default_x0({"neurons":params_generators, "synapses":params_synapses})
    with h5py.File(FILE4SAVING, "r") as dfile:
        X = dfile["X"][:]
else:
    X, bounds, x_names = get_default_x0({"neurons": params_generators, "synapses": params_synapses})
#print(X.size, len(bounds))

# timer = time.time()
# Erev_hist, tau_m_hist = simulate(X, Duration, dt, Cm, animal_velocity, params_generators, params_synapses)
# print(time.time() - timer)

args = (Duration, dt, Cm, animal_velocity, params_generators, params_synapses, target_params)
timer = time.time()

print('starting optimization ... ')
sol = differential_evolution(Loss, x0=X, popsize=32, atol=1e-3, recombination=0.7, \
                                 mutation=0.2, bounds=bounds, maxiter=500, \
                                 workers=-1, updating='deferred', disp=True, strategy='best2bin', \
                                 polish=True, args=args, callback=callback)
callback(sol)
print("Time of optimization ", time.time() - timer, " sec")
print("success ", sol.success)
print("message ", sol.message)
print("number of interation ", sol.nit)
X = sol.x

# with h5py.File("results.h5", "r") as dfile:
#     X = dfile["X"][:]
print("Optimal parameters:")
for idx in range(len(X)):
    print(x_names[idx], " = ", X[idx])

Erev_hist, tau_m_hist = simulate(X, Duration, dt, Cm, animal_velocity, params_generators, params_synapses)
t = np.arange(0, Duration, dt)
#t = t[100:]
tc = 0.5*Duration
Etarget = get_target_Esyn(t, tc, dt, theta_freq, animal_velocity, target_params)

fig, axes = plt.subplots(nrows=2)
axes[0].plot(t, Erev_hist)
axes[0].plot(t, Etarget)
axes[1].plot(t, tau_m_hist)


for ax in axes:
    ax.ticklabel_format(useOffset=False)
plt.show()

