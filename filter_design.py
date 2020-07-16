import numpy as np


def rcos_freq(f, beta, T, is_on_cuda=False):
    """Frequency response of a raised cosine filter with a given roll-off factor and width """
    if is_on_cuda:
        import cupy as np
    else:
        import numpy as np
    rc = np.zeros(f.shape[0], dtype=f.dtype)
    rc[np.where(np.abs(f) <= (1 - beta) / (2 * T))] = T
    idx = np.where((np.abs(f) > (1 - beta) / (2 * T)) & (np.abs(f) <= (
            1 + beta) / (2 * T)))
    rc[idx] = T / 2 * (1 + np.cos(np.pi * T / beta *
                                  (np.abs(f[idx]) - (1 - beta) /
                                   (2 * T))))
    return rc


def rrcos_freq(f, beta, T, is_on_cuda=False):
    """Frequency transfer function of the square-root-raised cosine filter with a given roll-off factor and time width/sampling period after _[1]
    Parameters
    ----------
    f   : array_like
        frequency vector
    beta : float
        roll-off factor needs to be between 0 and 1 (0 corresponds to a sinc pulse, square spectrum)
    T   : float
        symbol period
    Returns
    -------
    y   : array_like
       filter response
    References
    ----------
    ..[1] B.P. Lathi, Z. Ding Modern Digital and Analog Communication Systems
    """
    return np.sqrt(rcos_freq(f, beta, T, is_on_cuda))


def rrcos_pulseshaping_freq(sig, fs, T, beta, is_on_cuda=False):
    """
    Root-raised cosine filter in the spectral domain by multiplying the fft of the signal with the
    frequency response of the rrcos filter.
    Parameters
    ----------
    sig    : array_like
        input time distribution of the signal
    fs    : float
        sampling frequency of the signal
    T     : float
        width of the filter (typically this is the symbol period)
    beta  : float
        filter roll-off factor needs to be in range [0, 1]
    Returns
    -------
    sign_out : array_like
        filtered signal in time domain
    """
    if is_on_cuda:
        import cupy as np
    else:
        import numpy as np

    f = np.fft.fftfreq(sig.shape[0]) * fs
    nyq_fil = rrcos_freq(f, beta, T, is_on_cuda)
    nyq_fil /= nyq_fil.max()
    sig_f = np.fft.fft(sig)
    sig_out = np.fft.ifft(sig_f * nyq_fil)
    return sig_out


def ideal_lp(samples, left_freq, right_freq, fs, need_fft=True):
    if hasattr(samples, 'device'):
        import cupy as np
        from cupy.fft import fftfreq
    else:
        import numpy as np
        from scipy.fft import fftfreq

    if need_fft:
        if hasattr(samples, 'device'):
            from cupy.fft import fft, ifft
        else:
            from scipy.fft import fft, ifft
        samples = fft(samples)

    freq_vector = fftfreq(len(np.atleast_2d(samples)[0]), 1 / fs)

    mask1 = freq_vector <= left_freq

    mask2 = freq_vector > right_freq

    for row in samples:
        row[mask1] = 0
        row[mask2] = 0

    if need_fft:
        samples = ifft(samples, axis=-1)

    return samples

import scipy.fft as scifft
import scipy.signal as scisig
def filter_signal(signal, fs, cutoff, ftype="bessel", order=2, analog=False):
    """
    Apply an analog filter to a signal for simulating e.g. electrical bandwidth limitation
    Parameters
    ----------
    signal  : array_like
        input signal array
    fs      : float
        sampling frequency of the input signal
    cutoff  : float
        3 dB cutoff frequency of the filter
    ftype   : string, optional
        filter type can be either a bessel, butter, exp or gauss filter (default=bessel)
    order   : int
        order of the filter
    Returns
    -------
    signalout : array_like
        filtered output signal
    """
    sig = np.atleast_2d(signal)
    if ftype == "gauss":
        f = np.linspace(-fs/2, fs/2, sig.shape[1], endpoint=False, dtype=sig.dtype)
        w = cutoff/(2*np.sqrt(2*np.log(2))) # might need to add a factor of 2 here do we want FWHM or HWHM?
        g = np.exp(-f**2/(2*w**2))
        fsignal = scifft.fftshift(scifft.fft(scifft.fftshift(sig, axes=-1), axis=-1), axes=-1) * g
        if signal.ndim == 1:
            return scifft.fftshift(scifft.ifft(scifft.fftshift(fsignal))).flatten()
        else:
            return scifft.fftshift(scifft.ifft(scifft.fftshift(fsignal)))
    if ftype == "exp":
        f = np.linspace(-fs/2, fs/2, sig.shape[1], endpoint=False, dtype=sig.dtype)
        w = cutoff/(np.sqrt(2*np.log(2)**2)) # might need to add a factor of 2 here do we want FWHM or HWHM?
        g = np.exp(-np.sqrt((f**2/(2*w**2))))
        g /= g.max()
        fsignal = scifft.fftshift(scifft.fft(scifft.fftshift(signal))) * g
        if signal.ndim == 1:
            return scifft.fftshift(scifft.ifft(scifft.fftshift(fsignal))).flatten()
        else:
            return scifft.fftshift(scifft.ifft(scifft.fftshift(fsignal)))
    Wn = cutoff*2*np.pi if analog else cutoff
    frmt = "ba" if analog else "sos"
    fs_in = None if analog else fs
    if ftype == "bessel":
        system = scisig.bessel(order, Wn,  'low', norm='mag', analog=analog, output=frmt, fs=fs_in)
    elif ftype == "butter":
        system = scisig.butter(order, Wn, 'low',  analog=analog, output=frmt, fs=fs_in)
    if analog:
        t = np.arange(0, sig.shape[1])*1/fs
        sig2 = np.zeros_like(sig)
        for i in range(sig.shape[0]):
            to, yo, xo = scisig.lsim(system, sig[i], t)
            sig2[i] = yo.astype(sig.dtype)
    else:
        sig2 = scisig.sosfilt(system.astype(sig.dtype), sig, axis=-1)
    if signal.ndim == 1:
        return sig2.flatten()
    else:
        return sig2
