import numpy as np
import matplotlib.pyplot as plt
import slib as lib
import net_params as pr
import time
from copy import deepcopy
from scipy.optimize import differential_evolution
import h5py

###################################################################
def get_params():
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
    return params_generators, params_synapses, target_params
###################################################################
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
#############################################################################


params_generators, params_synapses, target_params = get_params()