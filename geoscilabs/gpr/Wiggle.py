import numpy as np


def PrimaryWave(x, velocity, tinterp):
    return tinterp + 1.0 / velocity * x


def ReflectedWave(x, velocity, tinterp):
    time = np.sqrt(x ** 2 / velocity ** 2 + tinterp ** 2)
    return time
