import numpy as np
import slib as lib
import net_params as pr
import matplotlib.pyplot as plt
def main():
    neurons_params = [pr.theta_generators, pr.neuron_params]
    neurons_idx_by_names = {}

    for neuron_idx, neuron in enumerate(neurons_params):
        neuron["class"] = getattr(lib, neuron["class"])
        neurons_idx_by_names[neuron["name"]] = neuron_idx

    synapses_params = []
    for synapse in pr.synapses_params:
        synapse_param = {}
        synapse_param["class"] = getattr(lib, synapse["class"])


        synapse_param["pre_idx"] = neurons_idx_by_names[synapse["pre_name"]]
        synapse_param["post_idx"] = neurons_idx_by_names[synapse["post_name"]]
        synapse_param["target_compartment"] = synapse["target_compartment"]


        for key in synapse["params"][0].keys():
            synapse_param[key] = np.zeros(len(synapse["params"]), dtype=np.float64 )

        for syn_idx, syn_el in enumerate(synapse["params"]):
            for key, val in syn_el.items():
                synapse_param[key][syn_idx] = val
        synapses_params.append(synapse_param)

    net = lib.Network(neurons_params, synapses_params)
    net.integrate(0.1, 1000)

    Vsoma = net.get_neuron_by_idx(1).getCompartmentByName('soma').getVhist()
    Vdend = net.get_neuron_by_idx(1).getCompartmentByName('dendrite').getVhist()

    firing = net.get_neuron_by_idx(1).getCompartmentByName('soma').getFiring() / 0.0001

    t = np.linspace(0, 1.0, Vsoma.size)

    fig,  axes = plt.subplots(nrows=2, sharex=True)
    axes[0].plot(t, Vsoma)
    axes[0].plot(t, Vdend)
    axes[1].plot(t, firing)
    plt.show()




if __name__ == '__main__':
    main()
    #test_generators()


