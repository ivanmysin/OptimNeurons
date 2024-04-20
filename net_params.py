import numpy as np

THETA_FREQ = 6.0 # Частота тета-ритма, Гц
V_AN = 10        # Скорость бега животного, cm/sec


default_param4optimization = {
    ### Spatial coding and physiological properties of hippocampal neurons in the Cornu Ammonis subregions ######
    "mean_firing_rate": 0.5,  # spikes / sec
    "R_place_cell": 0.2,
    "precession_slope" : 5.0, # cm/sec  # superfitial 5.0 # deep 10.0
    "precession_onset": 150.0, # deg # superfitial 150  # deep 220
    "sigma_place_field" : 8.5, # cm
    "peak_firing_rate": 8.0,  # spikes / sec
    "phase_out_place" : 180, # deg # superfitial 180 # deep 0.0
}

##### block of generators params #########
theta_generators = {
    "class" : "VonMissesGenerator",
    "name" : "out_place_ca1pyr",
    "params" : [
        {
            "name": "ca1pyr",
            "R": 0.2,
            "freq": THETA_FREQ,
            "mean_spike_rate": 0.5,
            "phase": 3.14,
        },
    ],
}

theta_spatial_generators_soma = {
    "class" : "VonMissesSpatialMolulation",
    "name" : "theta_spatial_inputs_soma",
    "params" : [
        {
            "name": "ca3pyr",
            "R": 0.3,
            "freq": THETA_FREQ,
            "mean_spike_rate": 0.5,
            "phase": 1.58,

            "sigma_sp": 8.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 8.0,  # spike sec in the center of field
            "sp_centers": 5.0,  # cm
        },
        {
            "name": "pvbas",
            "R": 0.35,
            "freq": THETA_FREQ,
            "mean_spike_rate": 24.0,
            "phase": 1.58,

            "sigma_sp": 8.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 21.0,  # spike sec in the center of field
            "sp_centers": 0.0,  # cm
        },
        {
            "name": "cckbas",
            "R": 0.35,
            "freq": THETA_FREQ,
            "mean_spike_rate": 9.0,
            "phase": -1.57,

            "sigma_sp": 8.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 8.0,  # spike sec in the center of field
            "sp_centers": 0.0,  # cm
        },
        {
            "name": "aac",
            "R": 0.35,
            "freq": THETA_FREQ,
            "mean_spike_rate": 29.0,
            "phase": 0.0,

            "sigma_sp": 8.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 27.0,  # spike sec in the center of field
            "sp_centers": 0.0,  # cm
        },
        {
            "name": "bis",
            "R": 0.35,
            "freq": THETA_FREQ,
            "mean_spike_rate": 27.0,
            "phase": 3.14,

            "sigma_sp": 8.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 24.0,  # spike sec in the center of field
            "sp_centers": 0.0,  # cm
        },
    ],
}


theta_spatial_generators_dend = {
    "class" : "VonMissesSpatialMolulation",
    "name" : "theta_spatial_inputs_dend",
    "params" : [
        {
            "name": "ec3",
            "R": 0.3,
            "freq": THETA_FREQ,
            "mean_spike_rate": 1.5,
            "phase": -1.57,

            "sigma_sp": 5.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 8.0,  # spike sec in the center of field
            "sp_centers": -5.0,  # cm
        },
        {
            "name": "ivy",
            "R": 0.35,
            "freq": THETA_FREQ,
            "mean_spike_rate": 4.0,
            "phase": -1.58,

            "sigma_sp": 8.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 3.0,  # spike sec in the center of field
            "sp_centers": 0.0,  # cm
        },
        {
            "name": "ngf",
            "R": 0.35,
            "freq": THETA_FREQ,
            "mean_spike_rate": 8.0,
            "phase": 0.0,

            "sigma_sp": 8.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 7.0,  # spike sec in the center of field
            "sp_centers": 0.0,  # cm
        },
        {
            "name": "olm",
            "R": 0.35,
            "freq": THETA_FREQ,
            "mean_spike_rate": 30.0,
            "phase": 3.14,

            "sigma_sp": 8.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 27.0,  # spike sec in the center of field
            "sp_centers": 0.0,  # cm
        },
    ],
}

######################################################################
NN = 100 # number of neurons

# neuron_params = {
#     "class" : "OriginCompartment",
#     "name"  : "ca1pyr",
#     "V0" : np.array([0.0, ]),
# }


neuron_params = {
    "class" : "LIF",
    "name"  : "ca1pyr",
    "V0": np.zeros(NN, dtype=np.float64) - 5.0,
    "Cm": np.zeros(NN, dtype=np.float64) + 3.0,
    "Iextmean": 0.0, # np.zeros(1, dtype=np.float64) +
    "Iextvarience": 0.3, #
    "El": np.zeros(NN, dtype=np.float64),
    "gl": np.zeros(NN, dtype=np.float64) + 0.1,
    "Vt" : np.zeros(1, dtype=np.float64) + 20.0,
    "Vreset" : np.zeros(NN, dtype=np.float64) - 20.0,
}



# neuron_params = {
#     "class" : "ComplexNeuron",
#     "name"  : "ca1pyr",
#     "compartments" : [ {
#         "class" : "PyramideCA1Compartment",
#         "name" : "soma",
#         "V0": np.zeros(NN, dtype=np.float64) - 5.0,
#         "Cm": np.zeros(NN, dtype=np.float64) + 3.0,
#         "Iextmean": 0.0, # np.zeros(1, dtype=np.float64) +
#         "Iextvarience": 0.3, #
#         "ENa": np.zeros(NN, dtype=np.float64) + 120.0,
#         "EK": np.zeros(NN, dtype=np.float64) - 25.0,
#         "El": np.zeros(NN, dtype=np.float64) - 5.0,
#         "ECa": np.zeros(NN, dtype=np.float64) + 140.0,
#         "CCa": np.zeros(NN, dtype=np.float64) + 0.05,
#         "sfica": np.zeros(NN, dtype=np.float64) + 0.13,
#         "sbetaca": np.zeros(NN, dtype=np.float64) + 0.075,
#         "gbarNa": np.zeros(NN, dtype=np.float64) + 30.0,
#         "gbarK_DR": np.zeros(NN, dtype=np.float64) + 17.0,
#         "gbarK_AHP": np.zeros(NN, dtype=np.float64) + 0.8,
#         "gbarK_C ": np.zeros(NN, dtype=np.float64) + 15.0,
#         "gl": np.zeros(NN, dtype=np.float64) + 0.1,
#         "gbarCa": np.zeros(NN, dtype=np.float64) + 6.0,
#     },
#     {
#         "class": "PyramideCA1Compartment",
#         "name": "dendrite",
#         "V0": np.zeros(NN, dtype=np.float64) - 5.0,
#         "Cm": np.zeros(NN, dtype=np.float64) + 3.0,
#         "Iextmean": 0.0,
#         "Iextvarience": 0.3,
#         "ENa": np.zeros(NN, dtype=np.float64) + 120.0,
#         "EK": np.zeros(NN, dtype=np.float64) - 25.0,
#         "El": np.zeros(NN, dtype=np.float64) - 5.0,
#         "ECa": np.zeros(NN, dtype=np.float64) + 140.0,
#         "CCa": np.zeros(NN, dtype=np.float64) + 0.05,
#         "sfica": np.zeros(NN, dtype=np.float64) + 0.13,
#         "sbetaca": np.zeros(NN, dtype=np.float64) + 0.075,
#         "gbarNa": np.zeros(NN, dtype=np.float64) + 0.0,
#         "gbarK_DR": np.zeros(NN, dtype=np.float64) + 0.0,
#         "gbarK_AHP": np.zeros(NN, dtype=np.float64) + 0.8,
#         "gbarK_C ": np.zeros(NN, dtype=np.float64) + 5.0,
#         "gl": np.zeros(NN, dtype=np.float64) + 0.1,
#         "gbarCa": np.zeros(NN, dtype=np.float64) + 5.0,
#     }],
#     "connections" : [{
#         "compartment1": "soma",
#         "compartment2": "dendrite",
#         "p": np.array([0.5, ]),
#         "g": np.array([1.5, ]),
#     },],
# }



###############################################################################
### synapses block ############################################################
synapses_params = [
    {
        "class" : "PlasticSynapse",
        "pre_name": "out_place_ca1pyr",
        "post_name": "ca1pyr",
        "target_compartment" : "soma",
        "is_save_gsyn" : True,
        "params" : [
            {   # pyr to pyr connection
                "gmax" : 159341 * 1.310724564,
                "tau_d" : 6.489890385,
                "tau_r" : 801.5798994,
                "tau_f" : 19.0939326,
                "Uinc"  : 0.220334906,
                "Erev" : 60.0,
                "pconn" : 0.001,

                'gmax_nmda' : 10e3,  #0.0001 * 159341 * 0.1310724564,
                'Mg0' : 1.0,
                'b' : 3.57,
                'a_nmda' : 0.062,
                'tau_rise_nmda' : 2.0,
                'tau_decay_nmda' : 89.0,
            },
        ],
    },
    # {
    #     "class": "PlasticSynapse",
    #     "pre_name": "ca1pyr",
    #     "post_name": "ca1pyr",
    #     "target_compartment": "soma",
    #     "is_save_gsyn": True,
    #     "params": [
    #         {  # pyr to pyr connection
    #             "gmax": 100 * 1.310724564,
    #             "tau_d": 6.489890385,
    #             "tau_r": 801.5798994,
    #             "tau_f": 19.0939326,
    #             "Uinc": 0.220334906,
    #             "Erev": 60.0,
    #             "pconn": 0.1,
    #
    #             'gmax_nmda':  10e3, # 0.1310724564,  # 0.0001 * 159341 *
    #             'Mg0': 1.0,
    #             'b': 3.57,
    #             'a_nmda': 0.062,
    #             'tau_rise_nmda': 2.0,
    #             'tau_decay_nmda': 89.0,
    #         },
    #     ],
    # },
    {
        "class": "PlasticSynapse",
        "pre_name": "theta_spatial_inputs_soma",
        "post_name": "ca1pyr",
        "target_compartment": "soma",
        "is_save_gsyn": True,
        "params" : [
            {  # CA3 pyr to pyr connection
                "gmax": 75376 * 1.021220696,
                "tau_d": 7.463702539,
                "tau_r": 724.3667977,
                "tau_f": 18.01005789,
                "Uinc": 0.201847939,
                "Erev": 60.0,
                "pconn": 0.016, #0.16,

                'gmax_nmda': 10e3, #0.0001 *  75376 * 1.021220696,
                'Mg0': 1.0,
                'b': 3.57,
                'a_nmda': 0.062,
                'tau_rise_nmda': 2.0,
                'tau_decay_nmda': 89.0,
            },
            { # PV bas to pyr connection
                "gmax"  : 336 * 6.067811614,
                "tau_d" : 4.408643738,
                "tau_r" : 637.3779263,
                "tau_f" : 11.57699627,
                "Uinc"  : 0.282768383,
                "Erev"  : -15.0,
                "pconn" : 0.1, # 0.006,

                'gmax_nmda': 0.0,
                'Mg0': 1.0,
                'b': 3.57,
                'a_nmda': 0.062,
                'tau_rise_nmda': 2.0,
                'tau_decay_nmda': 89.0,
            },
            { # CCK bas to pyr connection
                "gmax"  : 395 * 1.633849863,
                "tau_d" : 8.261691664,
                "tau_r" : 659.1560802,
                "tau_f" : 55.80256764,
                "Uinc"  : 0.230934787,
                "Erev"  : -15.0,
                "pconn" : 0.1,  #0.006,

                'gmax_nmda': 0.0,
                'Mg0': 1.0,
                'b': 3.57,
                'a_nmda': 0.062,
                'tau_rise_nmda': 2.0,
                'tau_decay_nmda': 89.0,
            },
            { # Axo-Axonic cell to pyr connection
                "gmax"  : 312 * 3.059027201,
                "tau_d" : 6.288868745,
                "tau_r" : 700.9008886,
                "tau_f" : 8.885170901,
                "Uinc"  : 0.303356703,
                "Erev"  : -15.0,
                "pconn" : 0.1, # 0.006,

                'gmax_nmda': 0.0,
                'Mg0': 1.0,
                'b': 3.57,
                'a_nmda': 0.062,
                'tau_rise_nmda': 2.0,
                'tau_decay_nmda': 89.0,
            },
            { #CA1 Bistratified (-)0333
                "gmax"  : 804 * 1.4388733,
                "tau_d" : 11.2889288,
                "tau_r" : 1164.285493,
                "tau_f" : 9.208561312,
                "Uinc"  : 0.280258327,
                "Erev"  : -15.0,
                "pconn" : 0.1, # 0.007,

                'gmax_nmda': 0.0,
                'Mg0': 1.0,
                'b': 3.57,
                'a_nmda': 0.062,
                'tau_rise_nmda': 2.0,
                'tau_decay_nmda': 89.0,
            },
       ],
    },
    {
        "class": "PlasticSynapse",
        "pre_name": "theta_spatial_inputs_dend",
        "post_name": "ca1pyr",
        "target_compartment": "dendrite",
        "is_save_gsyn": True,
        "params": [
            {   # EC LIII Pyramidal
                "gmax"  : 22784 * 1.369309873,
                "tau_d" : 6.461721231,
                "tau_r" : 626.6211383,
                "tau_f" : 21.84321492,
                "Uinc"  : 0.236507156,
                "Erev"  : 60.0,
                "pconn" : 0.02, #0.008,

                'gmax_nmda': 10e3, #0.00001 * 22784 * 1.369309873,
                'Mg0': 1.0,
                'b': 3.57,
                'a_nmda': 0.062,
                'tau_rise_nmda': 2.0,
                'tau_decay_nmda': 89.0,
            },
            { # CA1 Ivy (-)0333
                "gmax"  : 778 * 1.372446563,
                "tau_d" : 10.09350595,
                "tau_r" : 994.5394996,
                "tau_f" : 11.0166117,
                "Uinc"  : 0.263139955,
                "Erev"  : -15.0,
                "pconn" : 0.1, # 0.006,

                'gmax_nmda': 0.0,
                'Mg0': 1.0,
                'b': 3.57,
                'a_nmda': 0.062,
                'tau_rise_nmda': 2.0,
                'tau_decay_nmda': 89.0,
            },
            {   # CA1 Neurogliaform (-)3000
                "gmax": 1114 * 1.645016607,
                "tau_d": 7.989148919,
                "tau_r": 783.1238783,
                "tau_f": 12.28169848,
                "Uinc": 0.233093825,
                "Erev": -15.0,
                "pconn": 0.1, # 0.002,

                'gmax_nmda': 0.0,
                'Mg0': 1.0,
                'b': 3.57,
                'a_nmda': 0.062,
                'tau_rise_nmda': 2.0,
                'tau_decay_nmda': 89.0,
            },
            {  # CA1 O-LM (-)1002
                "gmax"  : 521 * 1.645016607,
                "tau_d" : 7.989148919,
                "tau_r" : 783.1238783,
                "tau_f" : 12.28169848,
                "Uinc"  : 0.233093825,
                "Erev"  : -15.0,
                "pconn" : 0.1, # 0.002,

                'gmax_nmda': 0.0,
                'Mg0': 1.0,
                'b': 3.57,
                'a_nmda': 0.062,
                'tau_rise_nmda': 2.0,
                'tau_decay_nmda': 89.0,
            },
        ]
    },
]






################################################################################
