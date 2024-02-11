import pyximport
pyximport.install()
import numpy as np
import slib as lib
import net_params as pr
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time


x_params = []

class Simulator:
    def __init__(self):
        self.Duration = 1000 # ms 5000

        neurons_params = [pr.theta_generators, pr.theta_spatial_generators_soma, pr.theta_spatial_generators_dend,
                          pr.neuron_params]
        neurons_idx_by_names = {}

        for neuron_idx, neuron in enumerate(neurons_params):
            neuron["class"] = getattr(lib, neuron["class"])

            try:
                for neuron_param in neuron["params"]:
                    neuron_param["sp_centers"] += 0.5 * self.Duration*pr.V_AN/1000 # Distance
            except KeyError:  # либо 50005 либо 50000 получается - неправильный перевод
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
        
        for param in neurons_params:
            try:
                if param == pr.theta_generators:
                    x_params.append(param["params"][0]["mean_spike_rate"])
                else:
                    for i in range(len(param["params"])):
                        x_params.extend([param["params"][i]["sigma_sp"], 
                                        param["params"][i]["maxFiring"],
                                        param["params"][i]["sp_centers"] ]) 
            except KeyError: print("KeyError")

        #print(x_params)


    def run_model(self, X):

        self.neurons_params[0]["params"][0]["mean_spike_rate"] = X[0]
        c = 1
        try:
            for i in range(1, len(self.neurons_params)):
                for j in range(len(self.neurons_params[i]["params"])):
                    self.neurons_params[i]["params"][j]["sigma_sp"] = X[c]
                    self.neurons_params[i]["params"][j]["maxFiring"] = X[c+1]
                    self.neurons_params[i]["params"][j]["spcenters"] = X[c+2]
                    c+=3
        #self.synapses_params
        except KeyError: print("KeyError")
        print(self.neurons_params)

        net = lib.Network(self.neurons_params, self.synapses_params)
        net.integrate(0.1, self.Duration)

        Vsoma = net.get_neuron_by_idx(-1).getCompartmentByName('soma').getVhist()
        Vdend = net.get_neuron_by_idx(-1).getCompartmentByName('dendrite').getVhist()

        firing = net.get_neuron_by_idx(-1).getCompartmentByName('soma').getFiring() / 0.0001

        return firing
    
    


    def loss(self, X, sigma=0.15, center=5): 
        ################ Parameters for teor_spike_rate ##################
        slope = pr.default_param4optimization["default_param4optimization"] # deg/cm
        R = pr.theta_generators["params"][0]["R"]
        kappa = lib.r2kappa(R)
        theta_freq = pr.theta_generators["params"][0]["freq"] # 5 Hz
        animal_velocity = pr.V_AN  # cm/sec
        slope = animal_velocity * np.deg2rad(slope)
        ################ Parameters for teor_spike_rate ##################

        dt = 0.1 #ms
        t = np.arange(0, self.Duration, dt)

        teor_spike_rate = lib.get_teor_spike_rate(t, slope, theta_freq, kappa, sigma=0.15, center=5)
        simulated_spike_rate = self.run_model(X)
        
        L = np.mean(np.log((teor_spike_rate + 1) / (simulated_spike_rate + 1)) ** 2)
        # записать simulated_spike_rate, X, значение лосса - в hdf5
        return L

    def optimization(self):
    
        X = x_params # initial changable params

        bounds = [[0, 10]] # Boundaries for X
        for bnd_idx in range(len(X)-1):
            if bnd_idx % 3 == 0:
                bounds.append([0, 100]) # sigma_sp
            elif bnd_idx % 3 == 1:
                bounds.append([0, 100]) # maxFiring
            else:
                bounds.append([-100000, 100000]) # sp_centers


        timer = time.time()
        print('starting optimization ... ')


        loss_history = []
        x_history = []
        def callback(x):
            fobj = self.loss(x)
            loss_history.append(fobj)
            x_history.append(x)
            with open("results.txt", "w") as output:
                for i in range(len(loss_history)):
                    output.write('loss value: ' + str(loss_history[i]) + 'parameters:  ' + str(x_history[i]) + '\n')

        sol = differential_evolution(self.loss, x0=X, popsize=24, atol=1e-3, recombination=0.7, \
                                    mutation=0.2, bounds=bounds, callback=callback, maxiter=5, \
                                    workers=-1, updating='deferred', disp=True, \
                                    strategy='best2bin')
        X = sol.x

        print("Time of optimization ", time.time() - timer, " sec")
        print("success ", sol.success)
        print("message ", sol.message)
        print("number of interation ", sol.nit)

        return

   
def main():
    Duration = 5000 # Время симуляции в ms
    Distance = Duration * 0.001 * pr.V_AN  # Расстояние, которое пробегает животное за время симуляции в cm
    s = Simulator()
    s.optimization()

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



# def callbackF(Xi):
#     global Nfeval
#     print ('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], self.loss(Xi)))
#     Nfeval += 1

# print  ('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))  

# [xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = \
#     differential_evolution(self.loss, x0 = X, callback=callbackF, popsize=24, atol=1e-3, \
#             recombination=0.7, mutation=0.2, bounds=bounds, maxiter=10, full_output=True, \
#             retall=False, workers=-1, updating='deferred', strategy='best2bin')