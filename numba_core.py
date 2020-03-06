# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 03:52:17 2020

@author: shang
"""
import numba
import numpy as np

from numba import complex128,double
cma_core_type = \
    [(complex128[:, :], complex128[:, :], complex128[:, :], complex128[:, :], complex128[:, :], complex128[:, :], double)]

@numba.njit(cma_core_type, cache=True)
def cma_equalize_core(ex, ey, wxx, wyy, wxy, wyx, mu):
    # symbols = np.zeros((1,ex.shape[0]),dtype=np.complex128)
    symbols = np.zeros((2, ex.shape[0]), dtype=np.complex128)

    error_xpol_array = np.zeros((1, ex.shape[0]), dtype=np.float64)
    error_ypol_array = np.zeros((1, ex.shape[0]), dtype=np.float64)

    for idx in range(len(ex)):
        xx = ex[idx][::-1]
        yy = ey[idx][::-1]
        xout = np.sum(wxx * xx) + np.sum(wxy * yy)
        yout = np.sum(wyx * xx) + np.sum(wyy * yy)
        symbols[0, idx] = xout
        symbols[1, idx] = yout
        error_xpol = 1 - np.abs(xout) ** 2
        error_ypol = 1 - np.abs(yout) ** 2
        error_xpol_array[0, idx] = error_xpol
        error_xpol_array[0, idx] = error_ypol
        wxx = wxx + mu * error_xpol * xout * np.conj(xx)
        wxy = wxy + mu * error_xpol * xout * np.conj(yy)
        wyx = wyx + mu * error_ypol * yout * np.conj(xx)
        wyy = wyy + mu * error_ypol * yout * np.conj(yy)

    return symbols, wxx, wxy, wyx, wyy, error_xpol_array, error_ypol_array