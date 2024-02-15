import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import parzen
import sys
sys.path.append("../")
import slib as lib
import net_params as pr




params_generators = pr.theta_generators
params_generators["class"] = getattr(lib, params_generators["class"])

meanSR = params_generators['params'][0]["mean_spike_rate"]
t = np.arange(0, 5000, 0.1).reshape(-1, 1)
generators = lib.VonMissesGenerator(params_generators)
firings = generators.get_firing(t)
firings = firings[:, 0]

firings = firings / 0.0001


t = t.ravel()
tc = 2550


maxFiring = 25.0
amp = 2*(maxFiring - meanSR) / (meanSR + 1) # maxFiring / meanSR - 1 #  range [-1, inf]
sigma_spt = 250

print(meanSR)
multip = (1 + amp * np.exp( -0.5 * ((t - tc) / sigma_spt)**2))
firing_sp = multip * firings



plt.plot(t, firings, color="red", linewidth=2)
plt.plot(t, firing_sp, color="blue", linewidth=1)



plt.show()
