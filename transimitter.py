import numpy as np


from .tools import rescale_signal
def quantize_signal_New(sig_in, nbits=6, rescale_in=True, rescale_out=True):
    """
    Function so simulate limited resultion using DACs and ADCs, limit quantization error to (-delta/2,delta/2) and set
        decision threshold at mid-point between two quantization levels.
    Parameters:
        sig_in:            Input signal, numpy array, notice: input signal should be rescale to (-1,1)
        nbits:          Quantization resolution
        rescale_in:        Rescale input signal to (-1,1)
        rescale_out:       Rescale output signal to (-input_max_swing,input_max_swing)
    Returns:
        sig_out:        Output quantized waveform
    """
    # 2**nbits interval within (-1,1), output swing is (-1+delta/2,1-delta/2)
    # Create a 2D signal
    sig_in = np.atleast_2d(sig_in)
    npols = sig_in.shape[0]

    # Rescale to
    sig = np.zeros((npols,sig_in.shape[1]), dtype=sig_in.dtype)
    if rescale_in:
        for pol in range(npols):
            # notice: different pol may have different scale factor which cause power different between x and y -pol.
            sig[pol] = rescale_signal(sig_in[pol], swing=1)

    # Clipping exist if signal range is larger than (-1,1)
    swing = 2
    delta = swing/2**nbits
    levels_out = np.linspace(-1+delta/2, 1-delta/2, 2**nbits)
    levels_dec = levels_out + delta/2

    sig_out = np.zeros(sig.shape, dtype="complex")
    for pol in range(npols):
        sig_quant_re = levels_out[np.digitize(sig[pol].real, levels_dec[:-1], right=False)]
        sig_quant_im = levels_out[np.digitize(sig[pol].imag, levels_dec[:-1], right=False)]
        sig_out[pol] = sig_quant_re + 1j * sig_quant_im

    if not np.iscomplexobj(sig):
        sig_out = sig_out.real

    if rescale_out:
        max_swing = np.maximum(abs(sig_in.real).max(), abs(sig_in.imag).max())
        sig_out = sig_out * max_swing

    return sig_in.recreate_from_np_array(sig_out)

def modulator_response(rfsig_i,rfsig_q,dcsig_i=3.5,dcsig_q=3.5,dcsig_p=3.5/2,vpi_i=3.5,vpi_q=3.5,vpi_p=3.5,gi=1,gq=1,gp=1,ai=0,aq=0):
    """
    Function so simulate IQ modulator response.
    Parameters
    ----------
    rfsig_i:  array_like
            RF input signal to I channel
    rfsig_q:  array_like
            RF input signal to Q channel
    dcsig_i:  float
            DC bias signal to I channel
    dcsig_q:  float
            DC bias signal to Q channel
    dcsig_p:  float
            DC bias signal to outer MZM used to control the phase difference of I anc Q signal
            Normally is set to vpi_p/2, which correspond to 90 degree
    vpi_i: float
            Vpi of the MZM (zero-power point) in I channel
    vpi_q: float
            Vpi of the MZM (zero-power point) in Q channel
    vpi_p: float
            Vpi of the outer MZM (zero-power point) used to control phase difference.
    gi: float
            Account for split imbalance and path dependent losses of I MZM. i.e. gi=1 for ideal MZM with infinite extinction ratio
    gq: float
            Account for split imbalance and path dependent losses of Q MZM
    gp: float
            Account for split imbalance and path dependent losses of Q MZM
    ai: float
            Chirp factors of I channel MZM, caused by the asymmetry in the electrode design of the MZM. i.e. ai = 0 for ideal MZM
    aq: float
            Chirp factors of Q channel MZM, caused by the asymmetry in the electrode design of the MZM
    Returns
    -------
    e_out: array_like
            Output signal of IQ modulator. (i.e. Here assume that input laser power is 0 dBm)
    """

    volt_i = rfsig_i + dcsig_i
    volt_q = rfsig_q + dcsig_q
    volt_p = dcsig_p
    # Use the minus sign (-) to modulate lower level RF signal to corresponding Low-level optical field, if V_bias = Vpi
    e_i = -(np.exp(1j*np.pi*volt_i*(1+ai)/(2*vpi_i)) + gi*np.exp(-1j*np.pi*volt_i*(1-ai)/(2*vpi_i)))/(1+gi)
    e_q = -(np.exp(1j*np.pi*volt_q*(1+aq)/(2*vpi_q)) + gq*np.exp(-1j*np.pi*volt_q*(1-aq)/(2*vpi_q)))/(1+gq)
    e_out = np.exp(1j*np.pi/4)*(e_i*np.exp(-1j*np.pi*volt_p/(2*vpi_p)) + gp*e_q*np.exp(1j*np.pi*volt_p/(2*vpi_p)))/(1+gp)
    return e_out

def er_to_g(ext_rat):
    """
    Parameters
    ----------
    ext_rat
    Returns
    -------
    """
    g = (10**(ext_rat/20)-1)/(10**(ext_rat/20)+1)
    return g



from .tools import add_awgn
from .filter_design import filter_signal
def DAC_response(sig, enob, cutoff, fs,quantizer_model=True):
    """
    Function to simulate DAC response, including quantization noise (ENOB) and frequency response.
    Parameters
    ----------
    sig:              Input signal, signal object.
    enob:             Efficient number of bits      (i.e. 6 bits.)
    cutoff:           3-dB cutoff frequency of DAC. (i.e. 16 GHz.)
    quantizer_model:  if quantizer_model='true', use quantizer model to simulate quantization noise.
                      if quantizer_model='False', use AWGN model to simulate quantization noise.
    out_volt:         Targeted output amplitude of the RF signal.
    Returns
    -------
    filter_sig:     Quantized and filtered output signal
    snr_enob:       signal-to-noise-ratio induced by ENOB.
    """
    powsig_mean = (abs(sig) ** 2).mean()  # mean power of the real signal

    # Apply dac model to real signal
    if not np.iscomplexobj(sig):
        if quantizer_model:
            sig_enob_noise = quantize_signal_New(sig, nbits=enob, rescale_in=True, rescale_out=True)
            pownoise_mean = (abs(sig_enob_noise-sig)**2).mean()
            snr_enob = 10*np.log10(powsig_mean/pownoise_mean)  # unit:dB
        else:
            # Add AWGN noise due to ENOB
            x_max = abs(sig).max()           # maximum amplitude of the signal
            delta = x_max / 2**(enob-1)
            pownoise_mean = delta ** 2 / 12
            sig_enob_noise = add_awgn(sig, np.sqrt(pownoise_mean))
            snr_enob = 10*np.log10(powsig_mean/pownoise_mean)  # unit:dB

        # Apply 2-order bessel filter to simulate frequency response of DAC
        filter_sig = filter_signal(sig_enob_noise, fs, cutoff, ftype="bessel", order=2, analog=True)

    # Apply dac model to complex signal
    else:
        if quantizer_model:
            sig_enob_noise = quantize_signal_New(sig, nbits=enob, rescale_in=True, rescale_out=True)
            pownoise_mean = (abs(sig_enob_noise-sig)**2).mean()  # include noise in real part and imag part
            snr_enob = 10*np.log10(powsig_mean/pownoise_mean)  # unit:dB
        else:
            # Add AWGN noise due to ENOB

            x_max = np.maximum(abs(sig.real).max(), abs(sig.imag).max())    # maximum amplitude in real or imag part
            delta = x_max / 2**(enob-1)
            pownoise_mean = delta ** 2 / 12
            sig_enob_noise = add_awgn(sig, np.sqrt(2*pownoise_mean))  # add two-time noise power to complex signal
            snr_enob = 10*np.log10(powsig_mean/2/pownoise_mean)  # Use half of the signal power to calculate snr

        # Apply 2-order bessel filter to simulate frequency response of DAC
        filter_sig_re = filter_signal(sig_enob_noise.real, fs,cutoff, ftype="bessel", order=2)
        filter_sig_im = filter_signal(sig_enob_noise.imag, fs,cutoff, ftype="bessel", order=2)
        filter_sig = filter_sig_re + 1j* filter_sig_im

    return filter_sig, sig_enob_noise, snr_enob

def Simulate_transmitter_response(sig, fs,enob=6, cutoff=16e9, target_voltage=3.5, power_in=0):
    """
    Parameters
    ----------
    sig: array_like
            Input signal used for transmission
    enob: float
            efficient number of bits for DAC. UnitL bits
    cutoff: float
            3-dB cut-off frequency for DAC. Unit: GHz
    power_in: float
            Laser power input to IQ modulator. Default is set to 0 dBm.
    Returns
    -------
    """
    # Apply signal to DAC model
    [sig_dac_out, sig_enob_noise, snr_enob] = DAC_response(sig, enob, cutoff, fs,quantizer_model=False)

    # Amplify the signal to target voltage(V)
    sig_amp = ideal_amplifier_response(sig_dac_out, target_voltage)

    # Input quantized signal to IQ modulator
    rfsig_i = sig_amp.real
    rfsig_q = sig_amp.imag

    e_out = modulator_response(rfsig_i, rfsig_q, dcsig_i=3.5, dcsig_q=3.5, dcsig_p=3.5 / 2, vpi_i=3.5, vpi_q=3.5, vpi_p=3.5,
                       gi=1, gq=1, gp=1, ai=0, aq=0)
    power_out = 10 * np.log10( abs(e_out*np.conj(e_out)).mean() * (10 ** (power_in / 10)))

    # return e_out, power_out, snr_enob_i, snr_enob_q
    return e_out

def ideal_amplifier_response(sig,out_volt):
    """
    Simulate a ideal amplifier, which just scale RF signal to out_volt.
    Parameters
    ----------
    sig
    out_volt
    Returns
    -------
    """
    current_volt = max(abs(sig.real).max(), abs(sig.imag).max())
    return sig / current_volt * out_volt

