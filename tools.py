import numpy as np

import numpy as np


def upsampling(sig, up):
    sig = np.atleast_2d(sig[:])
    sig_new = np.zeros((sig.shape[0], sig.shape[1] * up), dtype=sig.dtype)

    for index, row in enumerate(sig_new):
        row[::up] = sig[index]

    return sig_new


def scatterplot(samples, sps=1):
    import matplotlib.pyplot as plt

    fignumber = samples.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=fignumber)
    for ith, ax in enumerate(axes):
        ax.scatter(samples[ith, ::sps].real, samples[ith, ::sps].imag, s=1, c='b')
        ax.set_aspect('equal', 'box')

        ax.set_xlim([samples[ith, ::sps].real.min() - 0.1, samples[ith, ::sps].real.max() + 0.1])
        ax.set_ylim(
            [samples[ith, ::sps].imag.min() - 0.1, samples[ith, ::sps].imag.max() + 0.1])

    plt.tight_layout()
    plt.show()


from scipy.constants import c


class Osa(object):

    def __init__(self, resnm, wavelength):
        '''
            wavelength: in m
        '''
        self.resnm = resnm
        self.resm = self.resnm * 1e-9
        self.wavelength = wavelength

    @property
    def reshz(self):
        reshz_ = c / (self.wavelength - self.resm / 2) - c / (self.wavelength + self.resm / 2)
        return reshz_

    def optical_spectrum(self, signal):
        from scipy.fft import fft, fftfreq, fftshift
        from scipy.signal import lfilter
        from scipy.interpolate import interp1d
        from scipy.constants import c
        signal.cpu()

        fs_in_fiber = signal.fs_in_fiber
        power = np.sum(np.abs(fft(signal[:], axis=-1)) ** 2, axis=0) / fs_in_fiber * self.reshz / len(signal)
        power = fftshift(power)

        df = fs_in_fiber / len(signal)

        windowlength = int(self.reshz / df)
        if divmod(windowlength, 2)[1] == 0:
            windowlength = windowlength + 1

        p = lfilter(np.ones((1, windowlength))[0] / windowlength, 1, power)

        p = np.roll(p, -(windowlength - 1) // 2)

        fc = signal.freq
        freq_vector = fftfreq(len(signal), 1 / signal.fs_in_fiber)
        freq_vector = fftshift(freq_vector)

        freq_vector_reshz = np.arange(np.min(freq_vector), np.max(freq_vector), self.reshz)

        y = interp1d(freq_vector, p)

        res = y(freq_vector_reshz)
        wavelength = c / (fc + freq_vector)
        wavelength = wavelength * 1e9

        res = 10 * np.log10(res * 1000)
        res = np.atleast_2d(res)
        wavelength = np.atleast_2d(wavelength)

        return res, wavelength

    def plot_osa(self, signal):
        import matplotlib.pyplot as plt
        res, wavelength = self.optical_spectrum(signal)
        res = np.atleast_2d(res)[0]
        wavelength = np.atleast_2d(wavelength)[0]

        plt.plot(wavelength, res)
        plt.xlabel(' wavelength [nm]', fontsize=16, fontname='Times New Roman')
        plt.ylabel(' Power [dBm]', fontsize=16, fontname="Times New Roman")
        plt.title(f"OSA,Resolution [{self.resnm} nm]")

    def estimate_osnr(self, signal, tol=1e-3):
        from scipy.interpolate import interp1d
        res, wavelength = self.optical_spectrum(signal)
        signal_noise_power = np.max(res[0])

        max_index = np.argmax(res[0])
        res = np.abs(res)

        idx1 = res[0, :max_index] < tol
        idx2 = res[0, max_index + 1:] < tol
        idx1 = np.nonzero(idx1)[0]
        idx2 = np.nonzero(idx2)[0] + max_index + 1

        noise = interp1d([res[0, idx1], res[0, idx2]], [wavelength[0, idx1], wavelength[0, idx2]])
        noise = noise(max_index)

        signal_noise_power = 10 ** (signal_noise_power / 10)
        noise_power = 10 ** (noise / 10)

        snr = (signal_noise_power - noise_power) / noise_power
        snr = 10 * np.log10(snr)

        osnr = self.convert_2osnr(snr, signal.wavelength)
        return osnr

    def convert_2osnr(self, snr, wavelength):
        res = snr + 10 * np.log10(self.reshz(wavelength) / 12.5e9)
        return res


def normalize(signal):
    samples = signal[:]
    samples = samples/np.sqrt(np.mean(np.abs(signal[:])**2,axis=-1,keepdims=True))
    return samples


def power_meter(signal):
    samples = signal[:]
    power = np.mean(np.abs(samples)**2,axis=-1,keepdims=True)
    power_dbm = 10*np.log10(power * 1000)
    return power, power_dbm


def calc_qam_ber_theory(qam_order,osnr,signal_bandwidth,is_pdm):
    from scipy.special import erfc
    def qfunc(x):
        y = 0.5 * erfc(x / np.sqrt(2))
        return y

    if is_pdm:
        esn0 = (osnr - 3) + 10 * np.log10((2 * 12.5e9) / signal_bandwidth)
        print(esn0)

    else:

        esn0 = osnr + 10 * 10*np.log10((2 * 12.5e9) / signal_bandwidth)

    esn0 = 10 ** (esn0 / 10)
    x = 2 * (1 - 1/np.sqrt(qam_order)) * qfunc( np.sqrt(3/(qam_order-1)*esn0) )
    symbolErrorRate = 1 - (1 - x)** 2

    bitErrorRateTheoritical = symbolErrorRate / np.log2(qam_order)
    return bitErrorRateTheoritical

def rescale_signal(E,swing=1):
    """
    Rescale the (1-pol) signal to (-swing, swing).
    """
    if np.iscomplexobj(E):
        scale_factor = np.maximum(abs(E.real).max(), abs(E.imag).max())
        return E / scale_factor * swing

    if not np.iscomplexobj(E):
        scale_factor = abs(E).max()
        return E / scale_factor * swing

def add_awgn(sig, strgth):
    """
    Add additive white Gaussian noise to a signal.
    Parameters
    ----------
    sig    : array_like
        signal input array can be 1d or 2d, if 2d noise will be added to every dimension
    strgth : float
        the strength of the noise to be added to each dimension
    Returns
    -------
    sigout : array_like
       output signal with added noise
    """
    return sig + (strgth * (np.random.randn(*sig.shape) + 1.j*np.random.randn(*sig.shape))/np.sqrt(2)).astype(sig.dtype) # sqrt(2) because of var vs std


def add_dispersion(sig, fs, D, L, wl0=1550):
    """
    Add dispersion to signal.
    Parameters
    ----------
    sig : array_like
        input signal
    fs : flaot
        sampling frequency of the signal (in SI units)
    D : float
        Dispersion factor in s/nm/km
    L : float
        Length of the dispersion in km
    wl0 : float,optional
        center wavelength of the signal
    Returns
    -------
    sig_out : array_like
        dispersed signal
    """
    from scipy.constants import c
    sig = np.atleast_2d(sig)
    N = sig.shape[-1]
    omega = np.fft.fftfreq(N, 1/fs)*np.pi*2
    beta2 = D * (wl0 * 1e-12) ** 2 / 2 / np.pi / c / 1e-3
    H = np.exp(-0.5j * omega**2 * beta2 * L)

    for index,row in enumerate(sig):
        row = np.fft.fft(np.fft.ifftshift(row, axes=-1), axis=-1)
        sig[index] = np.fft.fftshift(np.fft.ifft(row*H))
    return sig

if __name__ == '__main__':

    qam_order = 16
    snr = 15.45
    print(calc_qam_ber_theory(qam_order, snr, 35e9, 1))





