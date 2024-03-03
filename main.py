import pyximport
pyximport.install()
import numpy as np
import slib as lib
import net_params as pr
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.signal.windows import parzen
import time
import h5py
from pprint import pprint
from copy import deepcopy


class Simulator:
    def __init__(self, dt, duration, slope, animal_velocity, theta_freq, Rpc, sigma, params):
        self.Duration = duration # ms 5000
        self.dt = dt
        self.slope = slope
        self.animal_velocity = animal_velocity
        self.theta_freq = theta_freq
        self.Rpc = Rpc
        self.sigma = sigma

        neurons_params = deepcopy(params["neurons"])
        neurons_idx_by_names = {}

        for neuron_idx, neuron in enumerate(neurons_params):
            neuron["class"] = getattr(lib, neuron["class"])

            try:
                for neuron_param in neuron["params"]:
                    neuron_param["sp_centers"] += 0.5 * self.Duration*self.animal_velocity/1000 # Distance
            except KeyError:  # либо 50005 либо 50000 получается - неправильный перевод
                pass

            neurons_idx_by_names[neuron["name"]] = neuron_idx

        synapses_params = []
        for synapse in params["synapses"]:
            synapse_param = {}
            synapse_param["class"] = getattr(lib, synapse["class"])
            synapse_param["pre_idx"] = neurons_idx_by_names[synapse["pre_name"]]
            synapse_param["post_idx"] = neurons_idx_by_names[synapse["post_name"]]
            synapse_param["target_compartment"] = synapse["target_compartment"]

            for key in synapse["params"][0].keys():
                synapse_param[key] = np.zeros(len(synapse["params"]), dtype=np.float64)
            # в первом словаре списка params берутся ключи, и добавляются в новый synapse_param 
            # со значениями - списком нулей длины списка словарей params
            # каждый словарь списка params отвечает за отдельную пару соединяемых типов нейронов 
            for syn_idx, syn_el in enumerate(synapse["params"]): 
                # для каждого номера и словаря в списке params
                for key, val in syn_el.items():
                    synapse_param[key][syn_idx] = val
            synapses_params.append(synapse_param)

        self.neurons_params = neurons_params
        self.synapses_params = synapses_params


    def run_model(self, X):

        x_idx = 0

        #print( self.neurons_params[0]["params"][0]["maxFiring"] )

        for i in range(len(self.neurons_params)):
            try:
                for j in range(len(self.neurons_params[i]["params"])):
                    if self.neurons_params[i]["params"][j]["maxFiring"] < 0:
                        print(self.neurons_params[i]["params"][j]["maxFiring"])

                    self.neurons_params[i]["params"][j]["maxFiring"] = X[x_idx]
                    x_idx += 1

                    self.neurons_params[i]["params"][j]["sp_centers"] = X[x_idx] + 0.5 * self.Duration*self.animal_velocity/1000
                    x_idx += 1

                    self.neurons_params[i]["params"][j]["sigma_sp"] = X[x_idx]
                    x_idx += 1

            except KeyError:
                continue

        for synapse_type in self.synapses_params:
             for syn_idx in range(synapse_type["gmax"].size):
                 synapse_type["gmax"][syn_idx] = X[x_idx]
                 x_idx += 1




        net = lib.Network(self.neurons_params, self.synapses_params)
        net.integrate(self.dt, self.Duration)

        # Vsoma = net.get_neuron_by_idx(-1).getCompartmentByName('soma').getVhist()
        # Vdend = net.get_neuron_by_idx(-1).getCompartmentByName('dendrite').getVhist()

        firing = net.get_neuron_by_idx(-1).getCompartmentByName('soma').getFiring() # / (1000 * self.dt)

        win = parzen(101)
        win = win / np.sum(win)

        firing = np.convolve(firing, win, mode='same')

        return firing
    
    def r2kappa(self, R):
        """
        recalulate kappa from R for von Misses function
        """
        if R < 0.53:
            kappa = 2 * R + R ** 3 + 5 / 6 * R ** 5

        elif R >= 0.53 and R < 0.85:
            kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)

        elif R >= 0.85:
            kappa = 1 / (3 * R - 4 * R ** 2 + R ** 3)
        # kappa = np.where(R < 0.53,  2 * R + R ** 3 + 5 / 6 * R ** 5, 0.0)
        # kappa = np.where(np.logical_and(R >= 0.53, R < 0.85),  -0.4 + 1.39 * R + 0.43 / (1 - R), kappa)
        # kappa = np.where(R >= 0.85,  1 / (3 * R - 4 * R ** 2 + R ** 3), kappa)
        return kappa

    def get_teor_spike_rate(self, t, slope, theta_freq, kappa, sigma=1, center=0):
        teor_spike_rate = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        precession = 0.001 * t * slope

        phi0 = -2 * np.pi * theta_freq * 0.001 * center - np.pi - precession[np.argmax(teor_spike_rate)]
        teor_spike_rate *= np.exp(kappa * np.cos(2 * np.pi * theta_freq * t * 0.001 + precession + phi0))

        teor_spike_rate *= 0.001 # !!!!!!
        return teor_spike_rate

    def loss(self, X):
        ################ Parameters for teor_spike_rate ##################
        kappa = self.r2kappa(self.Rpc)
        slope = self.animal_velocity * np.deg2rad(self.slope)
        ################ Parameters for teor_spike_rate ##################

        t = np.arange(0, self.Duration, self.dt)
        center = 0.5*self.Duration

        sigma = self.sigma / self.animal_velocity * 1000

        teor_spike_rate = self.get_teor_spike_rate(t, slope, self.theta_freq, kappa, sigma=sigma, center=center)
        simulated_spike_rate = self.run_model(X)

        # fig, axes = plt.subplots()
        # cos_ref = 0.25*(np.cos(2*np.pi*t*0.001*self.theta_freq) + 1)
        # axes.plot(t, teor_spike_rate, color='red', label="Target firing rate")
        # axes.plot(t, cos_ref, linestyle='dashed')
        # plt.show()

        L = np.sum(np.log((teor_spike_rate + 1) / (simulated_spike_rate + 1)) ** 2)
        # записать simulated_spike_rate, X, значение лосса - в hdf5
        return L


def Loss(X, dt, duration, slope, animal_velocity, theta_freq, Rpc, sigma, params):

    s = Simulator(dt, duration, slope, animal_velocity, theta_freq, Rpc, sigma, params)

    loss = s.loss(X)

    return loss


def callback(res_obj, convergence):
    with h5py.File("results.h5", "w") as output:
        output.create_dataset("loss", data=res_obj.fun)
        output.create_dataset("X", data=res_obj.x)



def main():
    dt = 0.1 # ms
    Duration = 5000 # Время симуляции в ms

    Rpc = pr.default_param4optimization["R_place_cell"]
    theta_freq = pr.THETA_FREQ  # 5 Hz
    slope = pr.default_param4optimization["precession_slope"]  # deg/cm
    animal_velocity = pr.V_AN  # cm/sec
    sigma = pr.default_param4optimization["sigma_place_field"] # cm

    Distance = Duration * 0.001 * animal_velocity  # Расстояние, которое пробегает животное за время симуляции в cm

    if Distance < 8 * sigma:
        print("Расстояние, которое пробегает животное за время симуляции, меньше ПОЛЯ МЕСТА!!!")


    params = {
        "neurons" : [pr.theta_generators, pr.theta_spatial_generators_soma, pr.theta_spatial_generators_dend, pr.neuron_params],
        "synapses" : pr.synapses_params,
    }

    # initial changable params
    X0 = np.zeros(38, dtype=np.float64)
    bounds = []  # Boundaries for X

    x0_idx = 0
    for neurons_types in params["neurons"]:
        try:
            for neuron in neurons_types["params"]:
                X0[x0_idx] = neuron["maxFiring"]
                bounds.append([0.0001, 100])
                x0_idx += 1

                X0[x0_idx] = neuron["sp_centers"]
                bounds.append([-100000, 100000])
                x0_idx += 1

                X0[x0_idx] = neuron["sigma_sp"]
                bounds.append([0.1, 1000])
                x0_idx += 1
        except KeyError:
            continue

    for synapse_type in params["synapses"]:
        for syn in synapse_type["params"]:
            X0[x0_idx] = syn["gmax"]
            bounds.append([0.1, 10000])
            x0_idx += 1


    args = (dt, Duration, slope, animal_velocity, theta_freq, Rpc, sigma, params)
    #Loss(X0, *args)

    timer = time.time()
    print('starting optimization ... ')

    sol = differential_evolution(Loss, x0=X0, popsize=15, atol=1e-3, recombination=0.7, \
                                 mutation=0.2, bounds=bounds, callback=callback, maxiter=500, \
                                 workers=-1, updating='deferred', disp=True, strategy='best2bin', \
                                 args = args )

    print("Time of optimization ", time.time() - timer, " sec")
    print("success ", sol.success)
    print("message ", sol.message)
    print("number of interation ", sol.nit)
    print(sol.x)

    return






if __name__ == '__main__':
    main()

