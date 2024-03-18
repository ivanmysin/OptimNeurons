import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import parzen
import sys
sys.path.append("../")
import slib as lib
import net_params as pr
from scipy.special import i0 as bessel_i0
from scipy.signal.windows import parzen



def r2kappa(R):
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
# params_generators = pr.theta_generators
# params_generators["class"] = getattr(lib, params_generators["class"])

def get_target_firing_rate(t, tc, dt, theta_freq, v_an, params):


    meanSR = params['mean_firing_rate']
    phase = np.deg2rad(params['phase_out_place'])
    kappa = r2kappa( params["R_place_cell"] )

    SLOPE = np.deg2rad(params['precession_slope'] * v_an * 0.001)  # rad / ms
    ONSET = np.deg2rad(params['precession_onset'])
    ALPHA = 5.0

    mult4time = 2 * np.pi * theta_freq * 0.001

    I0 = bessel_i0(kappa)
    normalizator = meanSR / I0 * 0.001 * dt



    maxFiring = 25.0
    amp = 2*(maxFiring - meanSR) / (meanSR + 1) # maxFiring / meanSR - 1 #  range [-1, inf]
    sigma_spt = 250

    #print(meanSR)
    multip = (1 + amp * np.exp( -0.5 * ((t - tc) / sigma_spt)**2))





    start_place = t - tc - 3*sigma_spt
    end_place = t - tc + 3*sigma_spt
    inplace = 0.25 *(1.0 - ( start_place / (ALPHA + np.abs(start_place)))) * (1.0 +  end_place/ (ALPHA + np.abs(end_place)))


    precession = SLOPE * t * inplace
    phases = phase * (1 - inplace) - ONSET * inplace

    firings = normalizator * np.exp(kappa * np.cos(mult4time * t + precession - phases) )

    firing_sp = multip * firings # / (0.001 * dt)

    return firing_sp


t = np.arange(0, 5000, 0.1) #.reshape(-1, 1)
tc = 2550
theta_freq = 6.0
dt = 0.1
V_AN = pr.V_AN

firing_sp = get_target_firing_rate(t, tc, dt, theta_freq, V_AN, pr.default_param4optimization) / (0.001 * dt)


cos_ref = 1.2*(np.cos(2*np.pi*t*theta_freq * 0.001) + 1)
plt.plot(t, cos_ref, color="green", linewidth=0.5)
plt.plot(t, firing_sp, color="blue", linewidth=2)



plt.show()
