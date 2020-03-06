import numpy as np


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


if __name__ == '__main__':

    qam_order = 16
    snr = 15.45
    print(calc_qam_ber_theory(qam_order, snr, 35e9, 1))





