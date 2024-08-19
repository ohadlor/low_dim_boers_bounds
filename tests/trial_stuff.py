import numpy as np


def run():
    a = np.array([0, 2, 4, 6])
    b = np.array([0, 1, -1])
    c = np.array([1, -1, 1, -1])

    ab = np.array(np.repeat(a, len(b), axis=0) + np.tile(b, len(a))).reshape((len(b), len(a)), order="F")
    print(c * ab)
    1 == 1
