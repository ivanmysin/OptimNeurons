import numpy as np
import slib as lib
import net_params as pr
import matplotlib.pyplot as plt

class Simulator:
    def __int__(self):
        neurons_params = [pr.theta_generators, pr.theta_spatial_generators_soma, pr.theta_spatial_generators_dend,
                          pr.neuron_params]
        neurons_idx_by_names = {}

        for neuron_idx, neuron in enumerate(neurons_params):
            neuron["class"] = getattr(lib, neuron["class"])

            try:
                for neuron_param in neuron["params"]:
                    neuron_param["sp_centers"] += 0.5 * Distance
            except KeyError:
                pass

            neurons_idx_by_names[neuron["name"]] = neuron_idx

        synapses_params = []
        for synapse in pr.synapses_params:
            synapse_param = {}
            synapse_param["class"] = getattr(lib, synapse["class"])

            synapse_param["pre_idx"] = neurons_idx_by_names[synapse["pre_name"]]
            synapse_param["post_idx"] = neurons_idx_by_names[synapse["post_name"]]
            synapse_param["target_compartment"] = synapse["target_compartment"]

            for key in synapse["params"][0].keys():
                synapse_param[key] = np.zeros(len(synapse["params"]), dtype=np.float64)

            for syn_idx, syn_el in enumerate(synapse["params"]):
                for key, val in syn_el.items():
                    synapse_param[key][syn_idx] = val
            synapses_params.append(synapse_param)

        self.neurons_params = neurons_params
        self.synapses_params = synapses_params
    def run_model(self, X):

        self.neurons_params[ ... ] = X[ .... ]
        self.synapses_params[...] = X[ ... ]

        net = lib.Network(self.neurons_params, self.synapses_params)
        net.integrate(0.1, Duration)

        Vsoma = net.get_neuron_by_idx(-1).getCompartmentByName('soma').getVhist()
        Vdend = net.get_neuron_by_idx(-1).getCompartmentByName('dendrite').getVhist()

        firing = net.get_neuron_by_idx(-1).getCompartmentByName('soma').getFiring() / 0.0001

        return firing


def loss(X, args):
    simulator = args["Simulator"]

    teor_spike_rate = get_teor_spike_rate(args)
    simulated_spike_rate = simulator.run_model(X)

    L = np.mean(np.log((teor_spike_rate + 1) / (simulated_spike_rate + 1)) ** 2)
    return L

def main():
    Duration = 5000 # Время симуляции в ms
    Distance = Duration * 0.001 * pr.V_AN  # Расстояние, которое пробегает животное за время симуляции в cm





    # t = np.linspace(0, 0.001*Duration, Vsoma.size)
    #
    # fig,  axes = plt.subplots(nrows=2, sharex=True)
    # axes[0].plot(t, Vsoma)
    # axes[0].plot(t, Vdend)
    # axes[1].plot(t, firing)
    # plt.show()




if __name__ == '__main__':
    main()
    #test_generators()


