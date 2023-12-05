import numpy as np
import slib as lib
# import pyximport
# pyximport.install()
#import generators_lib
import matplotlib.pyplot as plt
def main():
    N = 5000

    soma_params = {
        "V0": np.zeros(N, dtype=np.float64) - 5.0,
        "Cm": np.zeros(N, dtype=np.float64) + 3.0,
        "Iextmean": 1.5, # np.zeros(1, dtype=np.float64) +
        "Iextvarience": 3.0, # np.zeros(1, dtype=np.float64) +
        "ENa": np.zeros(N, dtype=np.float64) + 120.0,
        "EK": np.zeros(N, dtype=np.float64) - 15.0,
        "El": np.zeros(N, dtype=np.float64) - 5.0,
        "ECa": np.zeros(N, dtype=np.float64) + 140.0,
        "CCa": np.zeros(N, dtype=np.float64) + 0.05,
        "sfica": np.zeros(N, dtype=np.float64) + 0.13,
        "sbetaca": np.zeros(N, dtype=np.float64) + 0.075,
        "gbarNa": np.zeros(N, dtype=np.float64) + 30.0,
        "gbarK_DR": np.zeros(N, dtype=np.float64) + 17.0,
        "gbarK_AHP": np.zeros(N, dtype=np.float64) + 0.8,
        "gbarK_C ": np.zeros(N, dtype=np.float64) + 15.0,
        "gl": np.zeros(N, dtype=np.float64) + 0.1,
        "gbarCa": np.zeros(N, dtype=np.float64) + 6.0,

        # "input_conduntance": np.empty((0, 0), dtype=np.float32),
        # "conduntances_Erev": np.empty(0, dtype=np.float32),
    }

    #soma = lib.PyramideCA1Compartment(soma_params)

    dendrite_params = {
        "V0": np.zeros(N, dtype=np.float64) - 5.0,
        "Cm": np.zeros(N, dtype=np.float64) + 3.0,
        "Iextmean": 0.0, #np.zeros(1, dtype=np.float64) + 0.0,
        "Iextvarience": 3.0, #np.zeros(1, dtype=np.float64) + 3.0,
        "ENa": np.zeros(N, dtype=np.float64) + 120.0,
        "EK": np.zeros(N, dtype=np.float64) - 15.0,
        "El": np.zeros(N, dtype=np.float64) - 5.0,
        "ECa": np.zeros(N, dtype=np.float64)+ 140.0,
        "CCa": np.zeros(N, dtype=np.float64) + 0.05,
        "sfica": np.zeros(N, dtype=np.float64) + 0.13,
        "sbetaca": np.zeros(N, dtype=np.float64) + 0.075,
        "gbarNa": np.zeros(N, dtype=np.float64) + 0.0,
        "gbarK_DR": np.zeros(N, dtype=np.float64) + 0.0,
        "gbarK_AHP": np.zeros(N, dtype=np.float64) + 0.8,
        "gbarK_C ": np.zeros(N, dtype=np.float64) + 5.0,
        "gl": np.zeros(N, dtype=np.float64) + 0.1,
        "gbarCa": np.zeros(N, dtype=np.float64) + 5.0,

        # "input_conduntance": np.empty((0, 0), dtype=np.float32),  # np.zeros( (1, 10000), dtype=np.float32) + 0.05, #
        # "conduntances_Erev": np.empty(0, dtype=np.float32),  # np.zeros(1, dtype=np.float32) + 120, #
    }

    #dend = lib.PyramideCA1Compartment(dendrite_params)

    connection_params = {
        "compartment1": "soma",
        "compartment2": "dendrite",
        "p": np.array([0.5, ]),
        "g": np.array([1.5, ]),
    }

    neuron_params = {
        "type": "pyramide",
        "compartments": [{'soma' : soma_params}, {'dendrite' : dendrite_params}],
        "connections": [connection_params, ]
    }

    pyramidal_cell = lib.ComplexNeuron(neuron_params["compartments"], neuron_params["connections"])

    pyramidal_cell.integrate(0.1, 200)
    Vsoma = pyramidal_cell.getCompartmentByName('soma').getVhist()
    Vdend = pyramidal_cell.getCompartmentByName('dendrite').getVhist()

    firing = pyramidal_cell.getCompartmentByName('soma').getFiring() / 0.0001

    t = np.linspace(0, 0.2, Vsoma.size)

    fig,  axes = plt.subplots(nrows=2)
    axes[0].plot(t, Vsoma)
    axes[0].plot(t, Vdend)
    axes[1].plot(t, firing)
    plt.show()


def test_generators():
    ##### block of generators params #########
    ca3pyr_params = {
        "name": "ca3pyr",
        "R": 0.3,
        "freq": 7.0,
        "mean_spike_rate": 0.5,  # 5,
        "phase": 1.58,
    }

    ca1pyr_params = {
        "name": "ca1pyr",
        "R": 0.2,
        "freq": 7.0,
        "mean_spike_rate": 0.5,  # 5,
        "phase": 3.14,
    }

    ec3_params = {
        "name": "ec3",
        "R": 0.2,
        "freq": 7.0,
        "mean_spike_rate": 1.5,  # 5,
        "phase": -1.57,
    }

    params_generators = [ca3pyr_params, ca1pyr_params, ec3_params]

    t = np.arange(0, 1000, 0.1).reshape(-1, 1)
    generators = lib.VonMissesGenerator(params_generators)
    firings = generators.get_firing(t)

    plt.plot(t, firings)
    plt.show()


if __name__ == '__main__':
    #main()
    test_generators()


