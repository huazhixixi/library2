# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 03:52:17 2020

@author: shang
"""
import numba
import numpy as np

from numba import complex128,double
cma_core_type = \
    [(complex128[:, :], complex128[:, :], complex128[:, :], complex128[:, :], complex128[:, :], complex128[:, :], double,double)]

@numba.njit(cma_core_type, cache=True)
def cma_equalize_core(ex, ey, wxx, wyy, wxy, wyx, mu, reference_power):
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
        error_xpol = reference_power - np.abs(xout) ** 2
        error_ypol = reference_power - np.abs(yout) ** 2
        error_xpol_array[0, idx] = error_xpol
        error_xpol_array[0, idx] = error_ypol
        wxx = wxx + mu * error_xpol * xout * np.conj(xx)
        wxy = wxy + mu * error_xpol * xout * np.conj(yy)
        wyx = wyx + mu * error_ypol * yout * np.conj(xx)
        wyy = wyy + mu * error_ypol * yout * np.conj(yy)

    return symbols, wxx, wxy, wyx, wyy, error_xpol_array, error_ypol_array

@numba.njit(cache=True)
def lms_equalize_core(ex, ey, train_symbol,wxx, wyy, wxy, wyx, mu_train,mu_dd,is_train):
    symbols = np.zeros((2, ex.shape[0]), dtype=np.complex128)
    error_xpol_array = np.zeros((1, ex.shape[0]), dtype=np.float64)
    error_ypol_array = np.zeros((1, ey.shape[0]), dtype=np.float64)

    if is_train:
        train_symbol_xpol = train_symbol[0]
        train_symbol_ypol = train_symbol[1]

    for idx in range(len(ex)):
        xx = ex[idx][::-1]
        yy = ey[idx][::-1]
        xout = np.sum(wxx * xx) + np.sum(wxy * yy)
        yout = np.sum(wyx * xx) + np.sum(wyy * yy)
        symbols[0, idx] = xout
        symbols[1, idx] = yout

        if is_train:
            error_xpol = train_symbol_xpol[idx] - xout
            error_ypol = train_symbol_ypol[idx] - yout
        else:
            raise NotImplementedError
            # xpol_symbol = decision(xout, constl)
            # ypol_symbol = decision(yout, constl)
            # error_xpol = xout - xpol_symbol
            # error_ypol = yout - ypol_symbol

        error_xpol_array[0, idx] = np.abs(error_xpol)
        error_ypol_array[0, idx] = np.abs(error_ypol)
        if is_train:
            mu = mu_train
        else:
            mu = mu_dd

        wxx = wxx + mu * error_xpol * np.conj(xx)
        wxy = wxy + mu * error_xpol * np.conj(yy)
        wyx = wyx + mu * error_ypol * np.conj(xx)
        wyy = wyy + mu * error_ypol * np.conj(yy)

    return symbols, wxx, wxy, wyx, wyy, error_xpol_array, error_ypol_array
