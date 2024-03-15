import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import parzen
import sys
sys.path.append("../")
import slib as lib
import net_params as pr
from scipy.special import i0 as bessel_i0

dt = 0.1
V_AN = pr.V_AN
SLOPE = np.deg2rad( pr.default_param4optimization['precession_slope'] * pr.V_AN * 0.001) # rad / ms



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
params_generators = pr.theta_generators
params_generators["class"] = getattr(lib, params_generators["class"])

meanSR = params_generators['params'][0]["mean_spike_rate"]
phases = params_generators['params'][0]["phase"]
kappa = r2kappa(params_generators['params'][0]["R"])
mult4time = 2 * np.pi * params_generators['params'][0]["freq"] * 0.001

I0 = bessel_i0(kappa)
normalizator = meanSR / I0 * 0.001 * dt


t = np.arange(0, 5000, 0.1) #.reshape(-1, 1)
# generators = lib.VonMissesGenerator(params_generators)
# firings = generators.get_firing(t)




tc = 2550


maxFiring = 2.0
amp = 2*(maxFiring - meanSR) / (meanSR + 1) # maxFiring / meanSR - 1 #  range [-1, inf]
sigma_spt = 250

print(meanSR)
multip = (1 + amp * np.exp( -0.5 * ((t - tc) / sigma_spt)**2))



onset = 0.0
#precession = 0.001 * t * slope * 8

precession = SLOPE * t * ( np.abs(t - tc) < 3*sigma_spt)   #(multip > 2.5*meanSR)
plt.plot(t, precession)
plt.show()
firings = normalizator * np.exp(kappa * np.cos(mult4time * t + precession - phases - onset) )

firings = firings / (0.001 * dt)
firing_sp = multip * firings


cos_ref = 1.5*(np.cos(2*np.pi*t*params_generators['params'][0]["freq"] * 0.001) + 1)
plt.plot(t, cos_ref, color="green", linewidth=0.5)
plt.plot(t, firing_sp, color="blue", linewidth=2)



plt.show()
