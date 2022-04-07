import numpy as np


def convert_void_toC(voidnum):
    a = str(voidnum[0])
    b = str(voidnum[1])

    if b[0] == '-':
        con = a+b+"j"
    else:
        con = a+"+"+b+"j"

    return np.csingle(con)
