#-*- coding: utf-8 -*-
"""
lib full cython
"""

#STUFF = "Hi"
from libc.math cimport exp, cos, sqrt
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from cython.operator cimport dereference, preincrement
import numpy as np
cimport numpy as np
from libcpp.queue cimport queue
from cython.parallel cimport parallel, prange
cimport cython
from scipy.special import i0 as bessel_i0


cdef class OriginFiring:
    def __cinit__(self, params, dt=0.1):
        pass

    cpdef integrate(self, double dt, double duration):
        pass

    def getCompartmentByName(self, name):
        return self


cdef class OriginCompartment(OriginFiring):
    cdef np.ndarray V, Isyn, Iext, Icoms
    cdef np.ndarray Vhist
    cdef np.ndarray firing
    cdef np.ndarray g_tot, g_E

    def __cinit__(self, params):
        pass

    cdef np.ndarray getV(self):
        return self.V

    cdef void setIext(self, np.ndarray Iext):
        self.Iext = Iext

    cdef addIsyn(self, gsyn, gE):
        #self.Isyn += Isyn
        self.g_tot += gsyn
        self.g_E += gE

    cdef addIcoms(self, np.ndarray Icoms):
        self.Icoms += Icoms


    def getVhist(self):
        return self.Vhist

    def getLFP(self):
        return self.LFP

    def getFiring(self):
        return self.firing

    def get_firing(self):
        firing = 0
        if self.firing.size > 0:
            firing = self.firing[-1]
        return firing


    def getCompartmentsNames(self):
        return ["soma"]

    def getCompartmentByName(self, name):
        return self

    cpdef checkFired(self):
       pass

####################
cdef class PyramideCA1Compartment(OriginCompartment):

    cdef np.ndarray Cm, ENa, EK, El, ECa, CCa, sfica, sbetaca
    cdef np.ndarray gbarNa, gbarK_DR, gbarK_AHP, gbarK_C, gl, gbarCa
    cdef np.ndarray th
    cdef np.ndarray countSp
    cdef np.ndarray m, h, n, s, c, q
    cdef np.ndarray INa, IK_DR, IK_AHP, IK_C, ICa, Il

    cdef double  Iextmean, Iextvarience,
    #cdef np.ndarray input_conduntance, conduntances_Erev
    #cdef int conduntance_counter


    def __cinit__(self, params):
        self.V = params["V0"]
        self.Cm = params["Cm"]

        self.Iextmean = params["Iextmean"]
        self.Iextvarience = params["Iextvarience"]

        self.ENa = params["ENa"]

        ##print(self.ENa)

        self.EK = params["EK"]
        self.El = params["El"]
        self.ECa = params["ECa"]

        self.CCa = params["CCa"]
        self.sfica = params["sfica"]
        self.sbetaca = params["sbetaca"]

        self.gbarNa = params["gbarNa"]
        self.gbarK_DR = params["gbarK_DR"]

        self.gbarK_AHP = params["gbarK_AHP"]
        self.gbarK_C = params["gbarK_C "]

        self.gl = params["gl"]
        self.gbarCa = params["gbarCa"]

        #self.input_conduntance = params["input_conduntance"]
        #self.conduntances_Erev = params["conduntances_Erev"]
        #self.conduntance_counter = 0

        self.Vhist = np.array([])
        # self.LFP = np.array([])


        self.firing = np.array([])
        self.th = self.El + 40

        self.Isyn = np.zeros_like(self.V)
        self.Icoms = np.zeros_like(self.V)

        self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
        self.h = self.alpha_h() / (self.alpha_h() + self.beta_h())
        self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())
        self.s = self.alpha_s() / (self.alpha_s() + self.beta_s())
        self.c = self.alpha_c() / (self.alpha_c() + self.beta_c())
        self.q = self.alpha_q() / (self.alpha_q() + self.beta_q())

        self.g_tot = np.zeros_like(self.V)
        self.g_E = np.zeros_like(self.V)

        self.calculate_currents()




    cdef void calculate_currents(self):

        self.g_tot[:] *= 0.0
        self.g_E[:] *= 0.0

        self.Il = self.gl * (self.El - self.V)

        self.g_tot += self.gl
        self.g_E += self.gl * self.El

        gNa = self.gbarNa * self.m * self.m * self.h
        self.INa = gNa * (self.ENa - self.V)
        self.g_tot += gNa
        self.g_E += gNa * self.ENa

        gKDR = self.gbarK_DR * self.n
        self.IK_DR = gKDR * (self.EK - self.V)
        self.g_tot += gKDR
        self.g_E += gKDR * self.EK

        gKAHP = self.gbarK_AHP * self.q
        self.IK_AHP = gKAHP * (self.EK - self.V)
        self.g_tot += gKAHP
        self.g_E += gKDR * self.EK

        gCa = self.gbarCa * self.s * self.s
        self.ICa = gCa * (self.ECa - self.V)
        self.g_tot += gCa
        self.g_E += gCa * self.ECa

        cdef np.ndarray tmp = np.minimum(1.0, self.CCa / 250.0)
        gKC = self.gbarK_C * self.c * tmp
        self.IK_C = gKC * (self.EK - self.V)
        self.g_tot += gKC
        self.g_E += gKC * self.EK

        self.Iext = np.random.normal(self.Iextmean, self.Iextvarience, self.V.size)

        self.Isyn[:] *= 0.0
        self.Icoms[:] *= 0.0


    cdef np.ndarray alpha_m(self):
        cdef np.ndarray x = 13.1 - self.V
        x[x == 0] = 0.01
        cdef np.ndarray alpha = 0.32 * x / (np.exp(0.25 * x) - 1)
        return alpha


    cdef np.ndarray beta_m(self):
        cdef np.ndarray x = self.V - 40.1
        x[x == 0] = 0.01
        cdef np.ndarray beta = 0.28 * x / (np.exp(0.2 * x) - 1)
        return beta

    cdef np.ndarray alpha_h(self):
        cdef np.ndarray alpha = 0.128 * np.exp((17 - self.V) / 18)
        return alpha

    cdef np.ndarray beta_h(self):
        cdef np.ndarray  x = 40 - self.V
        x[x == 0] = 0.00001
        cdef np.ndarray  beta = 4 / (np.exp(0.2 * x) + 1)
        return beta

    cdef np.ndarray alpha_n(self):
        cdef np.ndarray x = 35.1 - self.V
        x[x == 0] = 0.00001
        cdef np.ndarray alpha = 0.016 * x / (np.exp(0.2 * x) - 1)
        return alpha

    cdef np.ndarray beta_n(self):
        cdef np.ndarray beta = 0.25 * np.exp(0.5 - 0.025 * self.V)
        return beta

    cdef np.ndarray alpha_s(self):
        cdef np.ndarray x = self.V - 65
        cdef np.ndarray alpha = 1.6 / (1 + np.exp(-0.072 * x))
        return alpha

    cdef np.ndarray beta_s(self):
        cdef np.ndarray x = self.V - 51.1
        x[x == 0] = 0.00001
        cdef np.ndarray  beta = 0.02 * x / (np.exp(0.2 * x) - 1)
        return beta

    cdef np.ndarray alpha_c(self):
        cdef np.ndarray alpha = np.zeros_like(self.V)

        greather = self.V > 50

        alpha[greather] = 2 * np.exp((6.5 - self.V[greather])/27)
        less = np.logical_not(greather)
        alpha[less] = np.exp( ((self.V[less] - 10)/11) - ((self.V[less] - 6.5)/27) ) / 18.975

        return alpha

    cdef np.ndarray beta_c(self):
        cdef np.ndarray beta = np.zeros_like(self.V)
        less = self.V < 0
        beta[less] = 2 * np.exp((6.5 - self.V[less])/27) - self.alpha_c()[less]
        return beta

    cdef np.ndarray alpha_q(self):
        cdef np.ndarray alpha = 0.00002 * self.CCa
        alpha[alpha > 0.01] = 0.01
        return alpha

    cdef double beta_q(self):
        return 0.001


    cdef np.ndarray h_integrate(self, double dt):
        cdef np.ndarray h_0 = self.alpha_h() / (self.alpha_h() + self.beta_h())
        cdef np.ndarray tau_h = 1 / (self.alpha_h() + self.beta_h())
        return h_0 - (h_0 - self.h) * np.exp(-dt / tau_h)


    cdef np.ndarray n_integrate(self, double dt):
        cdef np.ndarray n_0 = self.alpha_n() / (self.alpha_n() + self.beta_n() )
        cdef np.ndarray tau_n = 1 / (self.alpha_n() + self.beta_n())
        return n_0 - (n_0 - self.n) * np.exp(-dt / tau_n)

    cdef np.ndarray s_integrate(self, double dt):
        cdef np.ndarray s_0 = self.alpha_s() / (self.alpha_s() + self.beta_s() )
        cdef np.ndarray tau_s = 1 / (self.alpha_s() + self.beta_s())
        return s_0 - (s_0 - self.s) * np.exp(-dt / tau_s)

    cdef np.ndarray c_integrate(self, double dt):
        cdef np.ndarray c_0 = self.alpha_c() / (self.alpha_c() + self.beta_c() )
        cdef np.ndarray tau_c = 1 / (self.alpha_c() + self.beta_c())
        return c_0 - (c_0 - self.c) * np.exp(-dt / tau_c)

    cdef np.ndarray q_integrate(self, double dt):
        cdef np.ndarray q_0 = self.alpha_q() / (self.alpha_q() + self.beta_q() )
        cdef np.ndarray tau_q = 1 / (self.alpha_q() + self.beta_q())
        return q_0 - (q_0 - self.q) * np.exp(-dt / tau_q)

    cdef np.ndarray CCa_integrate(self, double dt):
        cdef np.ndarray k1 = self.CCa
        cdef np.ndarray k2 = k1 + 0.5 * dt * (self.sfica * self.ICa - self.sbetaca * k1)
        cdef np.ndarray k3 = k2 + 0.5 * dt * (self.sfica * self.ICa - self.sbetaca * k2)
        cdef np.ndarray k4 = k1 + dt * (self.sfica * self.ICa - self.sbetaca * k1)
        return (k1 + 2*k2 + 2*k3 + k4) / 6

    cpdef integrate(self, double dt, double duration):
        cdef double t = 0

        while (t < duration):
            self.Vhist = np.append(self.Vhist, self.V[0])

            self.countSp = self.V < self.th
            #I = self.Il + self.INa + self.IK_DR + self.IK_AHP + self.IK_C + self.ICa + self.Isyn + self.Icoms + self.Iext/np.sqrt(dt)
            I = self.Icoms + self.Iext/np.sqrt(dt)

            tau_m = self.Cm / self.g_tot
            Vinf = self.g_E / self.g_tot

            #print(Vinf)
            self.V = Vinf - (Vinf - self.V) * np.exp(-dt / tau_m)

            self.V += dt * I / self.Cm

            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.h = self.h_integrate(dt)
            self.n = self.n_integrate(dt)
            self.s = self.s_integrate(dt)
            self.c = self.c_integrate(dt)
            self.q = self.q_integrate(dt)
            self.CCa = self.CCa_integrate(dt)

            self.calculate_currents()


            self.checkFired()

            t += dt


    def checkFired(self):
        fired = np.logical_and( (self.V >= self.th), self.countSp)
        firing = np.mean(fired)
        self.firing = np.append(self.firing, firing)

cdef class IntercompartmentConnection:
    cdef OriginCompartment comp1
    cdef OriginCompartment comp2
    cdef np.ndarray g, p
    def __cinit__(self, OriginCompartment comp1, OriginCompartment comp2, np.ndarray g, np.ndarray p):
        self.comp1 = comp1
        self.comp2 = comp2
        self.g = g
        self.p = p

    def activate(self):

        cdef np.ndarray Icomp1 = (self.g / self.p) * (self.comp2.getV() - self.comp1.getV() )
        cdef np.ndarray Icomp2 = (self.g/(1 - self.p)) * (self.comp1.getV() - self.comp2.getV())

        self.comp1.addIcoms(Icomp1)
        self.comp2.addIcoms(Icomp2)

cdef class ComplexNeuron:
    cdef dict compartments # map [string, OriginCompartment*] compartments
    cdef list connections # vector [IntercompartmentConnection*] connections

    def __cinit__(self, params, dt=0.1):
        self.compartments = dict()

        for comp in params["compartments"]:
            #key, value = comp.popitem()
            self.compartments[comp["name"]] = PyramideCA1Compartment(comp)


        self.connections = []
        for conn in params["connections"]:
            self.connections.append(IntercompartmentConnection(self.compartments[conn["compartment1"]], self.compartments[conn["compartment2"]], conn["g"], conn["p"]   ) )

    def getCompartmentsNames(self):
        return self.compartments.keys()

    def integrate(self, double dt, double duration):
        cdef double t = 0

        while(t < duration):
            for p in self.compartments.values():
                p.integrate(dt, dt)

            for c in self.connections:
                c.activate()

            t += dt

    def getCompartmentByName(self, name):
        return self.compartments[name]

################################################################################
cdef class VonMissesGenerator(OriginFiring):
    cdef np.ndarray kappa, mult4time, phases, normalizator
    cdef double dt, t

    def __cinit__(self, params, dt=0.1):
        self.dt = dt

        params = params["params"]

        cdef np.ndarray Rs = np.zeros(len(params), dtype=np.float64)
        cdef np.ndarray omegas = np.zeros_like(Rs)
        self.phases = np.zeros_like(Rs)
        cdef np.ndarray mean_spike_rates = np.zeros_like(Rs)


        for p_idx, params_el in enumerate(params):
            Rs[p_idx] = params_el["R"]
            omegas[p_idx] = params_el["freq"]
            self.phases[p_idx] = params_el["phase"]
            mean_spike_rates[p_idx] = params_el["mean_spike_rate"]

        self.kappa = self.r2kappa(Rs)

        self.mult4time = 2 * np.pi * omegas * 0.001

        cdef np.ndarray I0 = bessel_i0(self.kappa)
        self.normalizator = mean_spike_rates / I0 * 0.001 * self.dt # units: probability of spikes during dt

        self.t = 0.0



    cdef np.ndarray r2kappa(self, R):
        """
        recalulate kappa from R for von Misses function
        """
        # if R < 0.53:
        #     kappa = 2 * R + R ** 3 + 5 / 6 * R ** 5
        #
        # elif R >= 0.53 and R < 0.85:
        #     kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
        #
        # elif R >= 0.85:
        #     kappa = 1 / (3 * R - 4 * R ** 2 + R ** 3)
        kappa = np.where(R < 0.53,  2 * R + R ** 3 + 5 / 6 * R ** 5, 0.0)
        kappa = np.where(np.logical_and(R >= 0.53, R < 0.85),  -0.4 + 1.39 * R + 0.43 / (1 - R), kappa)
        kappa = np.where(R >= 0.85,  1 / (3 * R - 4 * R ** 2 + R ** 3), kappa)
        return kappa

    def get_firing(self, t=np.array([])):
        if t.size == 0:
            t = self.t

        cdef np.ndarray firings = self.normalizator * np.exp(self.kappa * np.cos(self.mult4time * t - self.phases) )
        self.t += self.dt
        return firings

    def integrate(self, dt, duration):
        self.t += dt

cdef class VonMissesSpatialMolulation(VonMissesGenerator):
    cdef np.ndarray Amps, sigma_t, t_centers

    def __cinit__(self, params, dt=0.1):

        params = params["params"]

        cdef np.ndarray sigma_sp = np.zeros(len(params), dtype=np.float64)
        cdef np.ndarray maxFiring = np.zeros_like(sigma_sp)
        cdef np.ndarray v_an = np.zeros_like(sigma_sp)
        cdef np.ndarray mean_spike_rates = np.zeros_like(sigma_sp)
        cdef np.ndarray sp_centers = np.zeros_like(sigma_sp)


        for p_idx, params_el in enumerate(params):
            sigma_sp[p_idx] = params_el["sigma_sp"]
            maxFiring[p_idx] = params_el["maxFiring"]
            v_an[p_idx] = 0.001 * params_el["v_an"]
            mean_spike_rates[p_idx] = params_el["mean_spike_rate"]
            sp_centers[p_idx] = params_el["sp_centers"]

        self.sigma_t = sigma_sp / v_an
        self.t_centers = sp_centers / v_an
        self.Amps = 2*(maxFiring - mean_spike_rates) / (mean_spike_rates + 1)


    def get_firing(self, t=np.array([])):
        if t.size == 0:
            t = self.t

        cdef np.ndarray phase_firings = self.normalizator * np.exp(self.kappa * np.cos(self.mult4time * t - self.phases) )
        cdef np.ndarray spatial_firings = (1 + self.Amps * np.exp( -0.5 * ((t - self.t_centers) / self.sigma_t)**2))

        cdef np.ndarray firings = phase_firings * spatial_firings
        return firings

##############################################################################################
cdef class BaseSynapse:
    cdef np.ndarray Erev, gmax, pconn, gsyn_hist
    cdef double dt
    cdef bool is_save_gsyn
    cdef OriginFiring presyn
    cdef OriginCompartment postsyn

    def __cinit__(self, params, dt=0.1, is_save_gsyn=False):
        self.Erev = np.asarray(params['Erev'])
        self.gmax = np.asarray(params['gmax'])
        self.pconn = np.asarray(params['pconn'])
        self.dt = dt
        self.is_save_gsyn = is_save_gsyn

        if self.is_save_gsyn:
            #self.gsyn_hist = np.zeros_like(self.gmax).reshape(-1, 1)
            self.gsyn_hist = np.zeros(4, dtype=np.float64).reshape(-1, 1)

    def set_presyn(self, presyn):
        self.presyn = presyn

    def set_postsyn(self, postsyn):
        self.postsyn = postsyn

    def get_gsyn_hist(self):
        return self.gsyn_hist

cdef class PlasticSynapse(BaseSynapse):
    cdef np.ndarray tau1r, tau_d, tau_r, tau_f, Uinc, X, U, R
    cdef np.ndarray Mg0_b, a_nmda, tau_rise_nmda, tau_decay_nmda, gmax_nmda, h_nmda, gnmda

    def __cinit__(self, params, dt=0.1, is_save_gsyn=False):
        self.dt = dt

        self.Erev = np.asarray(params['Erev'])
        #self.Erev = np.reshape(self.Erev, (-1, 1))


        self.gmax = np.asarray(params['gmax'])
        self.pconn = np.asarray(params['pconn'])

        self.tau_d = np.asarray(params['tau_d'])
        self.tau_r = np.asarray(params['tau_r'])
        self.tau_f = np.asarray(params['tau_f'])
        self.Uinc = np.asarray(params['Uinc'])

        self.tau1r = np.where(self.tau_d != self.tau_r,  self.tau_d / (self.tau_d - self.tau_r), 1e-13)

        self.X = np.ones_like(self.gmax)   #(len(params), dtype=np.float64)
        self.U = np.zeros_like(self.X)
        self.R = np.zeros_like(self.X)

        self.gmax_nmda = np.asarray(params['gmax_nmda'])
        self.Mg0_b = np.asarray(params['Mg0']) / np.asarray(params['b'])
        self.a_nmda = np.asarray(params['a_nmda'])
        self.tau_rise_nmda = np.asarray(params['tau_rise_nmda'])
        self.tau_decay_nmda = np.asarray(params['tau_decay_nmda'])

        self.gnmda = np.zeros_like(self.X)
        self.h_nmda = np.zeros_like(self.X)




    def get_R(self):
        return self.R

    def add_Isyn2Post(self):
        Vpost = self.postsyn.getV()

        # cooношение размерностей тут!!!
        g_nmda = (self.gmax_nmda * self.gnmda).reshape(-1, 1) / (1.0 + np.exp(-self.a_nmda.reshape(-1, 1) * (Vpost.reshape(1, -1) - 65.0) ) * self.Mg0_b.reshape(-1, 1) )

        g_nmda_tot = np.sum(g_nmda, axis=0)

        gsyn = self.gmax * self.R

        if self.is_save_gsyn:
            #self.gsyn_hist = np.append(self.gsyn_hist, gsyn.reshape(-1, 1), axis=1)
            self.gsyn_hist = np.append(self.gsyn_hist, g_nmda_tot.reshape(-1, 1), axis=1)

        #gsyn = np.reshape(gsyn, (-1, 1))

        #Vdiff = self.Erev - np.reshape(Vpost, (1, -1))
        #Itmp = gsyn * Vdiff
        #Isyn = np.sum(Itmp, axis=0)

        gE = np.sum(gsyn * self.Erev) #+ np.sum( g_nmda_tot * self.Erev.reshape(-1, 1), axis=0)
        gsyn_tot = np.sum(gsyn) #+ g_nmda_tot
        self.postsyn.addIsyn(gsyn_tot, gE)
        return

    def integrate(self, dt):
        SRpre = self.presyn.get_firing()
        Spre_normed = SRpre * self.pconn

        y_ = self.R * np.exp(-dt / self.tau_d)

        x_ = 1 + (self.X - 1 + self.tau1r * self.U) * np.exp(-dt / self.tau_r) - self.tau1r * self.U

        u_ = self.U * np.exp(-dt / self.tau_f)

        released_mediator = self.U * x_ * Spre_normed
        self.U = u_ + self.Uinc * (1 - u_) * Spre_normed
        self.R = y_ + released_mediator
        self.X = x_ - released_mediator

        self.h_nmda = self.h_nmda * np.exp(-dt / self.tau_rise_nmda) + released_mediator
        self.gnmda = self.gnmda * np.exp(-dt / self.tau_decay_nmda) + self.h_nmda
        self.add_Isyn2Post()
        return
################################################################################################
cdef class Network:
    cdef list neurons
    cdef list synapses
    #cdef list neuron_params, synapse_params
    cdef double t

    def __cinit__(self, neuron_params, synapse_params, dt=0.1):
        self.neurons = list()
        self.synapses = list()

        self.t = 0


        for neuron_param in neuron_params:
            neuron = neuron_param["class"](neuron_param, dt=dt)
            self.neurons.append(neuron)


        for syn_param in synapse_params:
            synapse = syn_param["class"](syn_param, dt=dt, is_save_gsyn=syn_param['is_save_gsyn'])
            synapse.set_presyn(self.neurons[syn_param["pre_idx"]].getCompartmentByName("soma"))

            synapse.set_postsyn(self.neurons[syn_param["post_idx"]].getCompartmentByName(syn_param["target_compartment"]))
            self.synapses.append(synapse)


    def integrate(self, double dt, double duration):
        cdef int NN = len(self.neurons)
        cdef int NS = len(self.synapses)
        cdef int s_ind = 0
        cdef int neuron_ind = 0
        cdef double t = 0
        while(t < duration):
            for neuron_ind in range(NN):
                #print(neuron_ind)
                self.neurons[neuron_ind].integrate(dt, dt)

            for s_ind in range(NS):
                self.synapses[s_ind].integrate(dt)

            t += dt
            self.t += dt

    def get_neuron_by_idx(self, idx):
        return self.neurons[idx]

    def get_synapse_by_idx(self, idx):
        return self.synapses[idx]