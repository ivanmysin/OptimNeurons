import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import parzen
import sys
sys.path.append("../")
import slib as lib
import net_params as pr


params_generators = pr.theta_generators # pr.theta_spatial_generators_soma  #
params_generators["class"] = getattr(lib, params_generators["class"])

duration = 1.0 # sec


t = np.arange(0, 1000*duration, 0.1).reshape(-1, 1)
generators = params_generators["class"](params_generators)
firings = generators.get_firing(t)
# firings = firings[:, 0]
#
# NN = 10000
# firings_MonteCarlo = np.zeros_like(firings)
# for idx, fired in enumerate(firings):
#     firings_MonteCarlo[idx] = np.mean( np.random.rand(NN) < fired )
#
# print( np.sum(firings_MonteCarlo) )
#
# win = parzen(101)
# win = win / np.sum(win)
# firings_MonteCarlo = np.convolve(firings_MonteCarlo, win, mode='same')



#plt.plot(t, firings_MonteCarlo, color="blue", linewidth=1)
plt.plot(t, firings, linewidth=1)
plt.show()

