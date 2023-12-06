import numpy as np
##### block of generators params #########
theta_generators = {
    "class" : "VonMissesGenerator",
    "name" : "theta_inputs",
    "params" : [
    {
        "name": "ca3pyr",
        "R": 0.3,
        "freq": 7.0,
        "mean_spike_rate": 0.5,
        "phase": 1.58,
    },
    {
        "name": "ca1pyr",
        "R": 0.2,
        "freq": 7.0,
        "mean_spike_rate": 0.5,
        "phase": 3.14,
    },
    ],
}
# {
#     "name": "ec3",
#     "R": 0.2,
#     "freq": 7.0,
#     "mean_spike_rate": 1.5,
#     "phase": -1.57,
# # },

######################################################################
NN = 2000 # number of neurons
neuron_params = {
    "class" : "ComplexNeuron",
    "name"  : "ca1pyr",
    "compartments" : [ {
        "class" : "PyramideCA1Compartment",
        "name" : "soma",
        "V0": np.zeros(NN, dtype=np.float64) - 5.0,
        "Cm": np.zeros(NN, dtype=np.float64) + 3.0,
        "Iextmean": 0.0, # np.zeros(1, dtype=np.float64) +
        "Iextvarience": 0.01, #0.3, # np.zeros(1, dtype=np.float64) +
        "ENa": np.zeros(NN, dtype=np.float64) + 120.0,
        "EK": np.zeros(NN, dtype=np.float64) - 15.0,
        "El": np.zeros(NN, dtype=np.float64) - 5.0,
        "ECa": np.zeros(NN, dtype=np.float64) + 140.0,
        "CCa": np.zeros(NN, dtype=np.float64) + 0.05,
        "sfica": np.zeros(NN, dtype=np.float64) + 0.13,
        "sbetaca": np.zeros(NN, dtype=np.float64) + 0.075,
        "gbarNa": np.zeros(NN, dtype=np.float64) + 30.0,
        "gbarK_DR": np.zeros(NN, dtype=np.float64) + 17.0,
        "gbarK_AHP": np.zeros(NN, dtype=np.float64) + 0.8,
        "gbarK_C ": np.zeros(NN, dtype=np.float64) + 15.0,
        "gl": np.zeros(NN, dtype=np.float64) + 0.1,
        "gbarCa": np.zeros(NN, dtype=np.float64) + 6.0,
    },
    {
        "class": "PyramideCA1Compartment",
        "name": "dendrite",
        "V0": np.zeros(NN, dtype=np.float64) - 5.0,
        "Cm": np.zeros(NN, dtype=np.float64) + 3.0,
        "Iextmean": 0.0,
        "Iextvarience": 0.001, #0.3
        "ENa": np.zeros(NN, dtype=np.float64) + 120.0,
        "EK": np.zeros(NN, dtype=np.float64) - 15.0,
        "El": np.zeros(NN, dtype=np.float64) - 5.0,
        "ECa": np.zeros(NN, dtype=np.float64) + 140.0,
        "CCa": np.zeros(NN, dtype=np.float64) + 0.05,
        "sfica": np.zeros(NN, dtype=np.float64) + 0.13,
        "sbetaca": np.zeros(NN, dtype=np.float64) + 0.075,
        "gbarNa": np.zeros(NN, dtype=np.float64) + 0.0,
        "gbarK_DR": np.zeros(NN, dtype=np.float64) + 0.0,
        "gbarK_AHP": np.zeros(NN, dtype=np.float64) + 0.8,
        "gbarK_C ": np.zeros(NN, dtype=np.float64) + 5.0,
        "gl": np.zeros(NN, dtype=np.float64) + 0.1,
        "gbarCa": np.zeros(NN, dtype=np.float64) + 5.0,
    }],
    "connections" : [{
        "compartment1": "soma",
        "compartment2": "dendrite",
        "p": np.array([0.5, ]),
        "g": np.array([1.5, ]),
    },],
}
###############################################################################
### synapses block ############################################################
synapses_params = [{
    "class" : "PlasticSynapse",
    "pre_name": "theta_inputs",
    "post_name": "ca1pyr",
    "target_compartment" : "soma",
    "params" : [
        {   # pyr to pyr connection
            "gmax" : 5 * 1.310724564,  # 1000000
            "tau_d" : 6.489890385,
            "tau_r" : 801.5798994,
            "tau_f" : 19.0939326,
            "Uinc"  : 0.220334906,
            "Erev" : 0.0,
            "pconn" : 50,  #0.1,
        },
        {  # CA3 pyr to pyr connection
            "gmax": 1 * 1.021220696,
            "tau_d": 7.463702539,
            "tau_r": 724.3667977,
            "tau_f": 18.01005789,
            "Uinc": 0.201847939,
            "Erev" : 0.0,
            "pconn" : 50,  #0.1,
        },

    ],
}, ]

################################################################################
