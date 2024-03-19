import numpy as np
from scipy.optimize import rosen, differential_evolution
#bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
# result = differential_evolution(rosen, bounds, disp=True, polish=False)
# print( result.x, result.fun )

def log_cosh(y_true, y_pred):
    x = y_pred - y_true
    y = np.mean(x + np.log(1 + np.exp(-2.0 * x)) - np.log(2.0))
    return y

def Loss(X, x, y_true):

    y_pred = X[0] * x + X[1]

    l = log_cosh(y_true, y_pred)

    return l


x = np.random.normal(0, 1, 100)
y = 2 * x + 0.5 + np.random.normal(0, 0.1, 100)
# import matplotlib.pyplot as plt
# plt.scatter(x , y)
# plt.show()

bounds = [(-100, 100), (-100, 100) ]
sol = differential_evolution(Loss, bounds, disp=True, polish=False, args=(x, y))
print( sol.fun )
print( sol.x )
