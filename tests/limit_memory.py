# using resource
import resource
import numpy as np
def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

limit_memory(int(10e9) )


a = np.zeros( int(10e7 / 8), dtype=np.float64)