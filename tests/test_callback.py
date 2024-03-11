# import numpy as np
# from scipy.optimize import differential_evolution
#
#
# def callback(intermediate_result=None):
#     print(intermediate_result.fun)
#     print(intermediate_result.x)
#
#     return True
# def loss(X):
#     l = np.sum(X)
#     return l
#
# bounds = [[0, 1] for _ in range(10)]
#
#
# res = differential_evolution(loss, bounds=bounds, x0=np.random.rand(10), callback=callback)
# print(res)

from multiprocessing.pool import ThreadPool
import threading
import time

def test_worker(i):
    # To ensure that the worker gives up control of the processor we sleep.
    # Otherwise, the same thread may be given all the tasks to process.
    time.sleep(.1)
    return threading.get_ident(), i

def real_worker(x):
    # return the argument squared and the id of the thread that did the work
    return x**2, threading.get_ident()

POOLSIZE = 5
with ThreadPool(POOLSIZE) as pool:
    # chunksize = 1 is critical to be sure that we have 1 task per thread:
    thread_dict = {result[0]: result[1]
                   for result in pool.map(test_worker, range(POOLSIZE), 1)}
    assert(len(thread_dict) == POOLSIZE)
    print(thread_dict)
    value, id = pool.apply(real_worker, (7,))
    print(value) # should be 49
    assert (id in thread_dict)
    print('thread index = ', thread_dict[id])