from .dsp_tools import _segment_axis
from .filter_design import rrcos_pulseshaping_freq
from .numba_core import cma_equalize_core
import numpy as np
import numba
import matplotlib.pyplot as plt
from .signal_define import Signal
from typing import List

def cd_compensation(span,signal:Signal,fs):
    '''
        span: The span for cd_c,should inlcude the following attributes:
            beta2:callable: receive the signal wavelength and return beta2
        signal:
            in place modify the signal

    '''
    if signal.is_on_cuda:
        import cupy as np
    else:
        import numpy as np

    center_wavelength = signal.wavelength
    freq_vector = np.fft.fftfreq(len(signal[0]),1/fs)
    omeg_vector = 2*np.pi*freq_vector
    if not isinstance(span,list):
        spans = [span]
    else:
        spans = span

    for span in spans:
        beta2 = -span.beta2(center_wavelength)
        dispersion = (-1j/2) * beta2 * omeg_vector**2 * span.length 
        for row in signal[:]:
            row[:] = np.fft.ifft(np.fft.fft(row) * np.exp(dispersion))

    return signal

    

from .signal_define import DummySignal,QamSignal

def matched_filter(signal, roll_off)->DummySignal:
    if signal.is_on_cuda:
        import cupy as np
    else:
        import numpy as np
    samples = np.copy(signal[:])
    for row in samples:
        row[:] = rrcos_pulseshaping_freq(row, signal.fs, 1 / signal.baudrate, roll_off, signal.is_on_cuda)
    
    return DummySignal(samples,signal.baudrate,signal.qam_order,signal.symbol,signal.is_on_cuda,signal.sps)



class Equalizer(object):
    def __init__(self,ntaps,lr,loops):
        self.wxx = np.zeros((1,ntaps),dtype = np.complex)
        self.wxy = np.zeros((1,ntaps),dtype = np.complex)

        self.wyx = np.zeros((1,ntaps),dtype = np.complex)

        self.wyy = np.zeros((1,ntaps),dtype = np.complex)

        self.wxx[0,ntaps//2] = 1
        self.wyy[0,ntaps//2] = 1
        
        self.ntaps = ntaps
        self.lr = lr
        self.loops = loops
        self.error_xpol_array = None
        self.error_ypol_array = None
        
        self.equalized_symbols = None
        
    def equalize(self,signal):
        
        raise NotImplementedError
        
    def scatterplot(self,sps=1):
        import matplotlib.pyplot as plt
        fignumber = self.equalized_symbols.shape[0]
        fig,axes = plt.subplots(nrows = 1,ncols = fignumber)
        for ith,ax in enumerate(axes):
            ax.scatter(self.equalized_symbols[ith,::sps].real,self.equalized_symbols[ith,::sps].imag,s=1,c='b')
            ax.set_aspect('equal', 'box')

            ax.set_xlim([self.equalized_symbols[ith,::sps].real.min()-0.1,self.equalized_symbols[ith,::sps].real.max()+0.1])
            ax.set_ylim([self.equalized_symbols[ith,::sps].imag.min()-0.1,self.equalized_symbols[ith,::sps].imag.max()+0.1])

        plt.tight_layout()
        plt.show()
    
    def plot_error(self):
        fignumber = self.equalized_symbols.shape[0]
        fig,axes = plt.subplots(figsize=(8,4),nrows = 1,ncols = fignumber)
        for ith,ax in enumerate(axes):
            ax.plot(self.error_xpol_array[0],c='b',lw=1)
        plt.tight_layout()
        plt.show()
        
    def plot_freq_response(self):
        from scipy.fftpack import fft,fftshift
        freq_res =  fftshift(fft(self.wxx)),fftshift(fft(self.wxy)),fftshift(fft(self.wyx)),fftshift(fft(self.wyy))
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(2,2)
        for idx,row in enumerate(axes.flatten()):
            row.plot(np.abs(freq_res[idx][0]))
            row.set_title(f"{['wxx','wxy','wyx','wyy'][idx]}")
        plt.tight_layout()
        plt.show()
        
    def freq_response(self):
        from scipy.fftpack import fft,fftshift
        freq_res =  fftshift(fft(self.wxx)),fftshift(fft(self.wxy)),fftshift(fft(self.wyx)),fftshift(fft(self.wyy))
        return freq_res

    
class CMA(Equalizer):
    
    
    def __init__(self,ntaps,lr,loops=3):
        super().__init__(ntaps,lr,loops)
       
        
    
    def equalize(self,signal):
        signal.cpu()
        import numpy as np
            
        samples_xpol = _segment_axis(signal[0],self.ntaps, self.ntaps-signal.sps)
        samples_ypol = _segment_axis(signal[1],self.ntaps, self.ntaps-signal.sps)
        
        self.error_xpol_array = np.zeros((self.loops,len(samples_xpol)))
        self.error_ypol_array = np.zeros((self.loops,len(samples_xpol)))
        
        for idx in range(self.loops):
            symbols, self.wxx, self.wxy, self.wyx, \
            self.wyy, error_xpol_array, error_ypol_array \
            = cma_equalize_core(samples_xpol,samples_ypol,\
                                self.wxx,self.wyy,self.wxy,self.wyx,self.lr)
            
            self.error_xpol_array[idx] = np.abs(error_xpol_array[0])**2
            self.error_ypol_array[idx] = np.abs(error_ypol_array[0])**2
        
        self.equalized_symbols = symbols

    
class PhaseRecovery(object):

    def prop(self,signal):
        raise NotImplementedError
        
    def plot_phase_noise(self):
        raise NotImplementedError
        
class Superscalar(PhaseRecovery):

    def __init__(self,block_length,g,filter_n,delay,pilot_number):
        '''
            block_length: the block length of the cpe
            g: paramater for pll
            filter_n: the filter taps of the ml
            pillot_number: the number of pilot symbols for each row
        '''
        self.block_length = block_length
        self.block_number = None
        self.g = g
        self.filter_n = filter_n
        self.delay = 0
        self.phase_noise = []
        self.cpr = []
        self.symbol_for_snr = []
        self.pilot_number = pilot_number
        self.const = None

    def prop(self,signal):
        self.const = signal.constl
        res,res_symbol = self.__divide_signal_into_block(signal)
        self.block_number = len(res[0])
        for row_samples,row_symbols in zip(res,res_symbol):
            phase_noise,cpr_temp,symbol_for_snr = self.__prop_one_pol(row_samples,row_symbols)
            self.cpr.append(cpr_temp)
            self.symbol_for_snr.append(symbol_for_snr)
            self.phase_noise.append(phase_noise)
        signal.samples = np.array(self.cpr)
        signal.symbol = np.array(self.symbol_for_snr)

        self.cpr = np.array(self.cpr)
        self.symbol_for_snr = np.array(self.symbol_for_snr)
        return signal

    def plot_phase_noise(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        for i in range(len(self.phase_noise)):
            axes = fig.add_subplot(1,len(self.phase_noise),i+1)
            axes.plot(self.phase_noise[i],lw=1,c='b')
        plt.show()


    def __divide_signal_into_block(self,signal):
        from .dsp_tools import _segment_axis
        res = []
        res_symbol = []
        for row in signal[:]:
            row = _segment_axis(row,self.block_length,0)
            res.append(row)

        for row in signal.symbol:
            row = _segment_axis(row, self.block_length, 0)
            res_symbol.append(row)

        for idx in range(len(res)):
            assert res[idx].shape == res_symbol[idx].shape
        if divmod(len(res[0]),2)[1]!=0:
            for idx in range(len(res)):
                res[idx] = res[idx][:-1,:]
                res_symbol[idx] = res_symbol[idx][:-1,::]

        return res, res_symbol

    def __prop_one_pol(self, row_samples, row_symbols):
        if divmod(len(row_samples),2)[1]!=0:
            row_samples = row_samples[:-1,:]
            row_symbols = row_symbols[:-1,:]
        ori_rx = row_samples.copy()
        ori_rx = ori_rx.reshape(-1)
        row_samples[::2,:] = row_samples[::2,::-1]
        row_symbols[::2,:] = row_symbols[::2,::-1]

        phase_angle_temp = np.mean(row_samples[::2,:self.pilot_number]/row_symbols[::2,:self.pilot_number],axis=-1,keepdims=True) \
                    + np.mean(row_samples[1::2,:self.pilot_number]/row_symbols[1::2,:self.pilot_number],axis=-1,keepdims=True)

        phase_angle_temp = np.angle(phase_angle_temp)
        # print(phase_angle_temp.shape)
        phase_angle = np.zeros((len(row_samples),1))
        phase_angle[::2] = phase_angle_temp
        phase_angle[1::2] = phase_angle_temp

        row_samples = row_samples * np.exp(-1j * phase_angle)

        cpr_symbols = self.parallel_pll(row_samples)

        cpr_symbols[::2,:] = cpr_symbols[::2,::-1]
        cpr_symbols.shape = 1,-1
        cpr_symbols = cpr_symbols[0]

        row_symbols[::2,:] = row_symbols[::2,::-1]
        row_symbols = row_symbols.reshape(-1)

        phase_noise = self.ml(cpr_symbols,ori_rx)
        # self.phase_noise = phase_angle
        # self.cpr = row_symbols * np.exp(-1j*self.phase_noise)


        return phase_noise,ori_rx * np.exp(-1j*phase_noise),row_symbols

    def ml(self,cpr,row_samples):
        from scipy.signal import lfilter
        decision_symbol = decision(cpr,self.const)
        h = row_samples/decision_symbol
        b = np.ones(2*self.filter_n + 1)
        h = lfilter(b,1,h,axis=-1)
        h = np.roll(h,-self.filter_n)
        phase = np.angle(h)
        return phase[0]


    def parallel_pll(self,samples):

        decision_symbols = samples
        cpr_symbols = samples.copy()
        phase = np.zeros(samples.shape)
        for ith_symbol in range(0,self.block_length-1):
            decision_symbols[:,ith_symbol] = decision(cpr_symbols[:,ith_symbol],self.const)
            tmp = cpr_symbols[:,ith_symbol]*np.conj(decision_symbols[:,ith_symbol])
            error = np.imag(tmp)
            phase[:,ith_symbol+1] = self.g * error + phase[:,ith_symbol]
            cpr_symbols[:,ith_symbol + 1]  = samples[:,ith_symbol + 1] * np.exp(-1j * phase[:,ith_symbol+1])

        return cpr_symbols





def decision(decision_symbols,const):
    decision_symbols = np.atleast_2d(decision_symbols)
    const = np.atleast_2d(const)[0]
    res = np.zeros_like(decision_symbols,dtype=np.complex128)
    for row_index,row in enumerate(decision_symbols):
        for index,symbol in enumerate(row):
            index_min = np.argmin(np.abs(symbol - const))
            res[row_index,index] = const[index_min]
    return res

