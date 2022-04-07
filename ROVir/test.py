import numpy as np
from numpy import linalg as LA
from scipy import signal

x = np.random.rand(3, 3)
I = np.identity(3)

print(f'{x=}')
print(f'{I=}')

print(f'{I.dot(x)=}')
