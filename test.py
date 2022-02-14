import numpy as np
from numpy import linalg as LA
from scipy import signal

x = np.random.rand(6, 8, 5)

a = signal.correlate(x[:, :, 1], x[:, :, 2], mode='valid')

print(f"{a=}")
print(f"{a.shape=}")
