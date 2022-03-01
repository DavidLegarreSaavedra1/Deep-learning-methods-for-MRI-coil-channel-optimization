import numpy as np
from numpy import linalg as LA
from scipy import signal

a = np.floor(
    np.random.rand(5, 5)
)

eigVal, eigVec = LA.eig(a)

print(f'{a.shape=}')

print(f'{eigVec.shape=}')
