import pyximport
pyximport.install()
import numpy as np
import slib as lib
import net_params as pr
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.signal.windows import parzen
from scipy.special import i0 as bessel_i0
import time
import h5py
from pprint import pprint
from copy import deepcopy

from multiprocessing.pool import Pool
import multiprocessing
import pickle

class Simulator:
    def __init__(self, dt, duration, animal_velocity, theta_freq, target_params,  params):
        self.Duration = duration # ms 5000
        self.dt = dt
        self.animal_velocity = animal_velocity
        self.theta_freq = theta_freq
        self.target_params = target_params




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
        synapses = deepcopy(params["synapses"])

        for synapse in synapses:
            synapse_param = {}
            synapse_param["class"] = getattr(lib, synapse["class"])
            synapse_param["pre_idx"] = neurons_idx_by_names[synapse["pre_name"]]
            synapse_param["post_idx"] = neurons_idx_by_names[synapse["post_name"]]
            synapse_param["target_compartment"] = synapse["target_compartment"]
            synapse_param["is_save_gsyn"] = synapse["is_save_gsyn"]

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


        #print( thread_idx )
        # if thread_idx == 1:
        #     for syn in synapses_params:
        #         pprint(syn)

        self.neurons_params = neurons_params
        self.synapses_params = synapses_params


    def run_model(self, X):

        x_idx = 0

        for i in range(len(self.neurons_params)):
            try:
                for j in range(len(self.neurons_params[i]["params"])):
                    if self.neurons_params[i]["params"][j]["maxFiring"] < 0:
                        print(self.neurons_params[i]["params"][j]["maxFiring"])

                    self.neurons_params[i]["params"][j]["maxFiring"] = X[x_idx]
                    x_idx += 1

                    self.neurons_params[i]["params"][j]["sp_centers"] = 1000*X[x_idx]/self.animal_velocity + 0.5 * self.Duration
                    x_idx += 1

                    self.neurons_params[i]["params"][j]["sigma_sp"] = X[x_idx]
                    x_idx += 1

            except KeyError:
                continue

        for synapse_type in self.synapses_params:
             for syn_idx in range(synapse_type["gmax"].size):
                 synapse_type["gmax"][syn_idx] = X[x_idx]
                 x_idx += 1

        # Устанавливаем NMDA проводимости
        for synapse_type in self.synapses_params:
             for syn_idx in range(synapse_type["gmax_nmda"].size):
                 if synapse_type["gmax_nmda"][syn_idx] == 0: continue
                 synapse_type["gmax_nmda"][syn_idx] = X[x_idx]
                 x_idx += 1


        try:
            thread_idx = int(multiprocessing.current_process()._identity[0])
        except IndexError:
            thread_idx = 0
        net = lib.Network(self.neurons_params, self.synapses_params, dt=self.dt)

        N_steps = int(self.Duration/self.dt)
        net.integrate(N_steps)
        # Vsoma = net.get_neuron_by_idx(-1).getCompartmentByName('soma').getVhist()
        # Vdend = net.get_neuron_by_idx(-1).getCompartmentByName('dendrite').getVhist()

        firing = net.get_neuron_by_idx(-1).getCompartmentByName('soma').getFiring() # / (1000 * self.dt)

        win = parzen(201)
        win = win / np.sum(win)

        firing = np.convolve(firing, win, mode='same')

        gE = 0.0
        gtot = 0.0

        for syn_idx, synapse in enumerate(self.synapses_params):
            if synapse['target_compartment'] == 'dendrite':
                continue
            gsyn = net.get_synapse_by_idx(syn_idx).get_gsyn_hist()

            gtot += np.sum(gsyn[:, 1:], axis=0)
            gE += np.sum(gsyn[:, 1:] * synapse['Erev'].reshape(-1, 1), axis=0)

        Erev_sum = gE / (gtot + 0.000001)


        return firing, Erev_sum
    
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

    def get_target_firing_rate(self, t, tc, dt, theta_freq, v_an, params):

        meanSR = params['mean_firing_rate']
        phase = np.deg2rad(params['phase_out_place'])
        kappa = self.r2kappa(params["R_place_cell"])

        SLOPE = np.deg2rad(params['precession_slope'] * v_an * 0.001)  # rad / ms
        ONSET = np.deg2rad(params['precession_onset'])
        ALPHA = 5.0

        mult4time = 2 * np.pi * theta_freq * 0.001

        I0 = bessel_i0(kappa)
        normalizator = meanSR / I0 * 0.001 * dt

        maxFiring = 25.0
        amp = 2 * (maxFiring - meanSR) / (meanSR + 1)  # maxFiring / meanSR - 1 #  range [-1, inf]
        sigma_spt = 250

        # print(meanSR)
        multip = (1 + amp * np.exp(-0.5 * ((t - tc) / sigma_spt) ** 2))

        start_place = t - tc - 3 * sigma_spt
        end_place = t - tc + 3 * sigma_spt
        inplace = 0.25 * (1.0 - (start_place / (ALPHA + np.abs(start_place)))) * (
                    1.0 + end_place / (ALPHA + np.abs(end_place)))

        precession = SLOPE * t * inplace
        phases = phase * (1 - inplace) - ONSET * inplace

        firings = normalizator * np.exp(kappa * np.cos(mult4time * t + precession - phases))

        firing_sp = multip * firings  # / (0.001 * dt)

        return firing_sp

    def log_cosh(self, y_true, y_pred):
        x = y_pred - y_true
        y = np.mean(x + np.log(1 + np.exp(-2.0 * x)) - np.log(2.0))

        return y

    def loss(self, X):
        ################ Parameters for teor_spike_rate ##################
        # kappa = self.r2kappa(self.Rpc)
        # slope = self.animal_velocity * np.deg2rad(self.slope)
        ################ Parameters for teor_spike_rate ##################

        t = np.arange(0, self.Duration, self.dt)
        center = 0.5*self.Duration

        sigma = self.target_params['sigma_place_field'] / self.animal_velocity * 1000



        teor_spike_rate = self.get_target_firing_rate(t, center, self.dt, self.theta_freq, self.animal_velocity, self.target_params)
        simulated_spike_rate, Erev_sum = self.run_model(X)

        E_tot_t = 40 * np.exp(-0.5 * ((t - 0.5 * t[-1]) / sigma) ** 2) #- 5.0
        #L = np.mean(np.log((teor_spike_rate + 1) / (simulated_spike_rate + 1)) ** 2)
        L =  self.log_cosh(teor_spike_rate, simulated_spike_rate)

        k = 0.01
        #L += k * np.mean( (E_tot_t - Erev_sum)**2 )
        L += k * self.log_cosh(E_tot_t, Erev_sum)

        
        #print("End loss")
        return L


def Loss(X, dt, duration, animal_velocity, theta_freq, target_params, params):

    s = Simulator(dt, duration, animal_velocity, theta_freq, target_params, params)

    loss = s.loss(X)

    return loss


def callback(intermediate_result=None):
    
    #print("Hello from callback!")
    with h5py.File("results.h5", "w") as output:
        output.create_dataset("loss", data=intermediate_result.fun)
        output.create_dataset("X", data=intermediate_result.x)

    return False

def main():
    dt = 0.1 # ms
    Duration = 5000 # Время симуляции в ms

    Rpc = pr.default_param4optimization["R_place_cell"]
    theta_freq = pr.THETA_FREQ  # 5 Hz
    target_params = pr.default_param4optimization  # deg/cm
    animal_velocity = pr.V_AN  # cm/sec


    Distance = Duration * 0.001 * animal_velocity  # Расстояние, которое пробегает животное за время симуляции в cm
    sigma = pr.default_param4optimization["sigma_place_field"]  # cm
    if Distance < 8 * sigma:
        print("Расстояние, которое пробегает животное за время симуляции, меньше ПОЛЯ МЕСТА!!!")


    params = {
        "neurons" : [pr.theta_generators, pr.theta_spatial_generators_soma, pr.theta_spatial_generators_dend, pr.neuron_params],
        "synapses" : pr.synapses_params,
    }

    # initial changable params
    X0 = np.zeros(42, dtype=np.float64)
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
            bounds.append([100, 1000000])
            x0_idx += 1

    # Устанавливаем мощности для NMDA
    for synapse_type in params["synapses"]:
        for syn in synapse_type["params"]:
            if syn["gmax_nmda"] == 0: continue
            X0[x0_idx] = syn["gmax_nmda"]
            bounds.append([1, 10e6])
            x0_idx += 1

    args = (dt, Duration, animal_velocity, theta_freq, target_params, params)

    # loss_p = (X0, ) + args
    #
    # with Pool(2) as p:
    #     p.starmap(Loss, ( loss_p, loss_p ))


    timer = time.time()
    print('starting optimization ... ')

    sol = differential_evolution(Loss, x0=X0, popsize=32, atol=1e-3, recombination=0.7, \
                                 mutation=0.2, bounds=bounds, maxiter=500, \
                                 workers=-1, updating='deferred', disp=True, strategy='best2bin', \
                                 polish=True, args = args, callback=callback)

    #sol = minimize(Loss, bounds=bounds, x0=X0, method='L-BFGS-B', args = args )
    callback(sol)
    print("Time of optimization ", time.time() - timer, " sec")
    print("success ", sol.success)
    print("message ", sol.message)
    print("number of interation ", sol.nit)
    print(sol.x)

    return






if __name__ == '__main__':
    main()

