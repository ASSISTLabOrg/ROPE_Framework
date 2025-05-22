"""
Contains miscellaneous operations and convenience functions.

Contact: Violet Player
Email: violet.player@noaa.gov
"""

#===================================== Imports =====================================#

import numpy as np
from typing import Union
from types import FunctionType as ftype

#### convencience
_array_like = Union[np.ndarray, list, tuple]

#===================================== Methods =====================================#

def vprint(statement, verbose=False):
    if verbose:
        print(statement)

def interpolate_matrix(x : _array_like,
                       xp : _array_like, 
                       A : np.ndarray, 
                       method="constant", 
                       axis=-1) -> np.ndarray:
    """
    Interpolates a 2D matrix along the requested axis

    Arguments:
        x : new axis of interpolation
        xp : original axis (same units as x)
        A : 2D array
        method : str, [constant, linear] interpolation
        axis : which of two axes to interpolate along?
    """

    ### array-ify
    x = np.array(x)
    xp = np.array(xp)

    ### build new matrix shape
    dim0, dim1 = A.shape
    if axis == 0:
        dim0 = len(x)
    else:
        dim1 = len(x)
    A_itp = np.zeros((dim0, dim1))

    #### iterate over other axis to interpolate
    for i in range(A_itp.shape[np.abs(axis) - 1]):
        if axis == 0:
            A_itp[:,i] = _interp_1D(x, xp, A[:,i], method)
        else:
            A_itp[i,:] = _interp_1D(x, xp, A[i,:], method)

    return A_itp

def _interp_1D(x, xp, yp, method="constant"):

    if method.lower() == "constant":
        y_itp = np.zeros(len(x))
        for i in range(len(x)):
            y_itp[i] = yp[np.argmin(np.abs(xp - x[i]))]

    elif method.lower() == "linear":
        y_itp = np.interp(x, xp, yp)

    else:
        raise Exception("Choose a valid interpolation method: [constant, linear]")
    
    return y_itp
