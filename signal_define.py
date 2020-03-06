import os

import matplotlib.pyplot as plt
import numpy as np

from .filter_design import rrcos_pulseshaping_freq
from .utilities import upsampling

BASE = os.path.dirname(os.path.abspath(__file__))

class Signal(object):

    def __init__(self, qam_order, baudrate, sps, sps_in_fiber, symbol_length, pol_number):
        '''
            qam_order
            message 2d-array
            all 2d-array

            baudrate:hz

        '''
        self.qam_order = qam_order
        self.message = None
        self.baudrate = baudrate
        self.sps = sps
        self.sps_in_fiber = sps_in_fiber
        self.ds = None
        self.ds_in_fiber = None
        self.symbol = None
        self.freq = None # center frequency
        self.symbol_length = symbol_length
        self.pol_number = pol_number
        self.__constl = None
        self.is_on_cuda = False

    @property
    def fs_in_fiber(self):
        return self.sps_in_fiber * self.baudrate

    def prepare(self, roll_off, is_cuda=False):
        raise NotImplementedError

    def __getitem__(self, value):
        return self.ds_in_fiber[value]

    def __setitem__(self, key, value):
        self.ds_in_fiber[key] = value

    @property
    def shape(self):
        return self.ds_in_fiber.shape

    def psd(self):
        if self.is_on_cuda:
            self.cpu()
            plt.figure()
            plt.psd(self.ds_in_fiber[0], NFFT=16384, Fs=self.fs_in_fiber, scale_by_freq=True)
            self.cuda()
            plt.show()
        else:
            plt.figure()
            plt.psd(self.ds_in_fiber[0], NFFT=16384, Fs=self.fs_in_fiber, scale_by_freq=True)
            plt.show()

    def __len__(self):
        if self.is_on_cuda:
            import cupy as np
        else:
            import numpy as np
        samples = np.atleast_2d(self[:])
        return len(samples[0])

    @property
    def constl(self):
        return self.__constl

    @constl.setter
    def constl(self, value):
        self.__constl = value

    def scatterplot(self, sps):
        
        flag = False
        if self.is_on_cuda:
            self.cpu()
            flag = True
            
            
        fignumber = self.shape[0]
        fig, axes = plt.subplots(nrows=1, ncols=fignumber)
        for ith, ax in enumerate(axes):
            ax.scatter(self.ds_in_fiber[ith, ::sps].real, self.ds_in_fiber[ith, ::sps].imag, s=1, c='b')
            ax.set_aspect('equal', 'box')

            ax.set_xlim(
                    [self.ds_in_fiber[ith, ::sps].real.min() - np.abs(self.ds_in_fiber[ith, ::sps].real.min())/3, self.ds_in_fiber[ith, ::sps].real.max() + np.abs(self.ds_in_fiber[ith, ::sps].real.max())/3])
            ax.set_ylim(
                    [self.ds_in_fiber[ith, ::sps].imag.min() - np.abs(self.ds_in_fiber[ith, ::sps].imag.min())/3, self.ds_in_fiber[ith, ::sps].imag.max() + np.abs(self.ds_in_fiber[ith, ::sps].imag.max())/3])

        plt.tight_layout()
        plt.show()
        
        if flag:
            self.cuda()
            

    @property
    def samples(self):
        return self.ds_in_fiber

    @samples.setter
    def samples(self, value):
        self.ds_in_fiber = value

    @property
    def fs(self):
        return self.baudrate * self.sps

    def cuda(self):
        if self.is_on_cuda:
            return
        try:
            import cupy as cp
        except ImportError:
            return
        
        self.ds_in_fiber = cp.array(self.ds_in_fiber)
        self.ds = cp.array(self.ds)
        self.is_on_cuda = True

        return self

    def cpu(self):
        if not self.is_on_cuda:
            return
        else:
            import cupy as cp
            self.ds_in_fiber = cp.asnumpy(self.ds_in_fiber)
            self.ds = cp.asnumpy(self.ds_in_fiber)
            self.is_on_cuda = False
        return self

    def save_to_mat(self,filename):
        from scipy.io import savemat
        flag = 0
        if self.is_on_cuda:
            self.cpu()
            flag = 1
        savemat(filename,dict(fs = self.fs,fs_in_fiber = self.fs_in_fiber,sps = self.sps,sps_in_fiber = self.sps_in_fiber,baudrate = self.baudrate,
                              samples_in_fiber = self.samples,symbol_tx = self.symbol))
        if flag:
            self.cuda()

    def save(self,file_name):
        flag = False
        if self.is_on_cuda:
            self.cpu()
            flat = True
        samples_in_fiber = self.ds_in_fiber
        samples = self.ds
        sps = self.sps
        sps_in_fiber = self.sps_in_fiber
        msg = self.message
        qam_order = self.qam_order
        baudrate = self.baudrate
        symbol = self.symbol
        freq = self.freq
        import joblib
        joblib.dump(dict(freq = freq,ds_in_fiber = samples_in_fiber,ds=samples,sps=sps,sps_in_fiber = sps_in_fiber,msg = msg,qamorder = qam_order,baudrate = baudrate,symbol = symbol,symbol_length = self.symbol_length,pol_number = self.pol_number,doinit = False),file_name)

        if flag:
            self.cuda()

    @classmethod
    def load(cls,filename):
        import joblib
        param = joblib.load(filename)
        signal = cls(**param)
        signal.samples = param['ds_in_fiber']
        signal.ds = param['ds']
        signal.message = param['msg']
        signal.symbol = param['symbol']
        signal.freq = param['freq']
        return signal


    @property
    def wavelength(self):
        from scipy.constants import c
        return c/self.freq
    
    @property
    def center_wavelength(self):
        return self.wavelength
    
    def inplace_normalise(self):
        np = self.get_module()
        factor = np.mean(np.abs(self[:])**2,axis=1,keepdims=True)
        self[:] = self[:]/np.sqrt(factor)
    
    def get_module(self):
        if self.is_on_cuda:
            import cupy as np
        else:
            import numpy as np

        return np

    def set_signal_power(self,power_in_dbm):
        np = self.get_module()
        self.inplace_normalise()
        power_linear = 10**(power_in_dbm/10)/1000/2
        self[:] = np.sqrt(power_linear) * self[:]

    @classmethod
    def load_mat(cls,array,sps,sps_in_fiber,tx_symbol,order,baudrate,msg = None,device='cpu'):
        tx_symbol = np.atleast_2d(tx_symbol)
        signal = cls(qamorder=order,baudrate = baudrate,sps = sps,sps_in_fiber=sps_in_fiber,symbol_length=tx_symbol.shape[1],pol_number=tx_symbol.shape[0])

        signal.ds = None
        signal.samples = None
        signal.message = None
        signal.symbol = None

        if msg:
            signal.message = msg

        signal.ds_in_fiber = array
        signal.symbol = tx_symbol
        signal.ds = signal.ds_in_fiber
        if device=='cuda':
            signal.cuda()

        return signal

    def to_32complex(self):
        if self.is_on_cuda:
            import cupy as np
        else:
            return
        self.ds_in_fiber = np.array(self.ds_in_fiber,dtype=np.complex64)
        self.ds = np.array(self.ds,dtype=np.complex64)
        self.is_on_cuda = True

class QamSignal(Signal):
    
    def __init__(self, qamorder, baudrate, sps, sps_in_fiber, symbol_length, pol_number,doinit = True,**kwargs):
        '''

        :param qamorder:
        :param baudrate: hz
        :param sps:
        :param sps_in_fiber:
        :param symbol_length:
        :param pol_number:
        '''
        super().__init__(qamorder, baudrate, sps, sps_in_fiber, symbol_length, pol_number)
        if doinit:
            self.message = np.random.randint(low=0, high=self.qam_order, size=(self.pol_number, self.symbol_length))
            self.map()

    def map(self):
        import joblib
        constl = joblib.load(BASE+'/constl')[self.qam_order][0]
        self.symbol = np.zeros_like(self.message, dtype=np.complex)
        for row_index, sym in enumerate(self.symbol):
            for i in range(self.qam_order):
                sym[self.message[row_index] == i] = constl[i]

        self.constl = constl

    def prepare(self, roll_off, is_cuda=False):

        self.ds = upsampling(self.symbol, self.sps)
        self.ds_in_fiber = np.zeros((self.pol_number, self.symbol.shape[1] * self.sps_in_fiber),
                                    dtype=self.symbol.dtype)
        if is_cuda:
            self.cuda()

        for index, row in enumerate(self.ds):
            row[:] = rrcos_pulseshaping_freq(row, self.fs, 1 / self.baudrate, roll_off, self.is_on_cuda)
            if not self.is_on_cuda:
                import resampy
                from scipy.signal import resample
                self.ds_in_fiber[index] = resampy.resample(row, self.sps, self.sps_in_fiber, filter='kaiser_fast')
            else:
                import cusignal
                self.ds_in_fiber[index] = cusignal.resample_poly(row, self.sps_in_fiber / self.sps, 1, axis=-1)
        # self.symbol[1] = self.symbol[0]
        # self.ds_in_fiber[1] = self.ds_in_fiber[0]
        return self

    @property
    def time_vector(self):
        return 1 / self.fs_in_fiber * np.arange(self.ds_in_fiber.shape[1])


class WdmSignal(object):
    
    def __init__(self, symbols, wdm_samples, freq, is_on_cuda, fs_in_fiber,center_freq,**kwargs):
        self.symbols = symbols
        self.wdm_samples = wdm_samples

        self.relative_freq = freq
        self.is_on_cuda = is_on_cuda
        self.fs_in_fiber = fs_in_fiber

        self.wdm_comb_config = None
        self.baudrates = None
        self.qam_orders = None
        self.center_freq = center_freq

        if self.is_on_cuda:
            import cupy as cp
            self.fs_in_fiber = cp.asnumpy(self.fs_in_fiber)
            self.wdm_comb_config = cp.asnumpy(self.wdm_comb_config)

            self.relative_freq = cp.asnumpy(self.relative_freq)

    def to_32complex(self):
        if self.is_on_cuda:
            import cupy as np
        else:
            import numpy as np

        wdm_samples = np.zeros_like(self.wdm_samples,dtype=np.complex64)
        wdm_samples[:] = self.wdm_samples[:]
        self.wdm_samples = wdm_samples

    def cuda(self):
        if self.is_on_cuda:
            return self
        else:
            try:
                import cupy as cp
            except ImportError:
                return self

            self.wdm_samples = cp.array(self.wdm_samples)
            self.is_on_cuda = True

        return self

    def cpu(self):
        if not self.is_on_cuda:
            return self
        else:
            import cupy as cp
            attris = dir(self)
            for attri in attris:
                if attri.startswith('__') and attri.endswith('__'):
                    continue
                else:
                    x = getattr(self,attri)
                    if isinstance(x,cp.ndarray):
                        x = cp.asnumpy(x)
                        setattr(self,attri,x)
            self.is_on_cuda = False
        return self

    def __getitem__(self, value):
        return self.wdm_samples[value]

    def __setitem__(self, key, value):
        self.wdm_samples[key] = value

    def psd(self):
        if self.is_on_cuda:
            self.cpu()

            plt.psd(self[0], NFFT=16384, Fs=self.fs_in_fiber, window=np.hamming(16384))
            plt.show()
            self.cuda()
        else:
            plt.psd(self[0], NFFT=16384, Fs=self.fs_in_fiber, window=np.hamming(16384))
            plt.show()

    @property
    def shape(self):
        return self.wdm_samples.shape


    def __len__(self):
        if self.is_on_cuda:
            import cupy as np
        else:
            import numpy as np
        return len(np.atleast_2d(self.wdm_samples)[0])

    def save_to_mat(self,filename):
        from scipy.io import savemat
        self.cpu()
        param = dict(fs_in_fiber = self.fs_in_fiber,wdm_samples = self.wdm_samples,

                     symbols = np.array(self.symbols),baudrates = np.array(self.baudrates),center_freq = self.center_freq,relative_freq = self.relative_freq
                     )
        savemat(filename,param)
        self.cuda()

    def save(self,file_name):
        self.cpu()

        param = dict( symbols = self.symbols ,
        wdm_samples = self.wdm_samples ,

        freq = self.relative_freq   ,     
        fs_in_fiber = self.fs_in_fiber ,     

        wdm_comb_config = self.wdm_comb_config  ,   
        baudrates = self.baudrates, 
        qam_orders = self.qam_orders,
        center_freq = self.center_freq
       )

        import joblib
        joblib.dump(param,file_name)
        self.cuda()

    @classmethod
    def load(cls,filename):
        import joblib
        param = joblib.load(filename)
        signal = cls(**param,is_on_cuda=False)
        signal.wdm_comb_config = param['wdm_comb_config']
        signal.baudrates = param['baudrates']
        signal.qam_orders = param['qam_orders']

        return signal

    @property
    def wavelength(self):
        from scipy.constants import  c
        return c/self.center_freq

    @property
    def length(self):
        if self.is_on_cuda:
            import cupy as np
        else:
            import numpy as np

        samples = np.atleast_2d(self.wdm_samples)
        length = samples.shape[1]
        return length

    @property
    def sps_in_fiber(self):
        return self.fs_in_fiber/self.baudrates[0]

class DummySignal:
    def __init__(self, samples, baudrate, qam_order, symbol, is_on_cuda, sps):
        self.samples = samples
        self.baudrate = baudrate
        self.qam_order = qam_order
        self.symbol = symbol
        self.is_on_cuda = is_on_cuda
        self.sps = sps
        
        if self.is_on_cuda:
            import cupy as cp
    @property
    def constl(self):
        import joblib
        constl = joblib.load(BASE + '/constl')[self.qam_order][0]
        return constl

    @property
    def fs(self):
        assert self.sps is not None
        return self.sps * self.baudrate

    def cpu(self):
        if not self.is_on_cuda:
            return
        else:
            import cupy as cp
            self.samples = cp.asnumpy(self.samples)
            self.is_on_cuda = False

    def cuda(self):
        if self.is_on_cuda:
            return
        else:
            try:
                import cupy as cp
            except ImportError:
                print('cuda not supported')
                return
            self.samples = cp.array(self.samples)
            self.is_on_cuda = True

    def __getitem__(self, key):
        return self.samples[key]

    def __setitem__(self,key,value):
        self.samples[key] = value


    def psd(self):
        if self.is_on_cuda:
            self.cpu()
            plt.psd(self[0], NFFT=16384, window=np.hamming(16384))
            plt.show()
            self.cuda()
        else:
            plt.psd(self[0], NFFT=16384, window=np.hamming(16384))
            plt.show()

    @property
    def shape(self):
        return self.samples.shape

    def scatterplot(self, sps):
        flag = False
        if self.is_on_cuda:
            self.cpu()
            flag = True
        fignumber = self.shape[0]
        fig, axes = plt.subplots(nrows=1, ncols=fignumber)
        for ith, ax in enumerate(axes):
            ax.scatter(self[ith, ::sps].real, self[ith, ::sps].imag, s=1, c='b')
            ax.set_aspect('equal', 'box')

            ax.set_xlim([self[ith, ::sps].real.min() - np.abs(self[ith, ::sps].real.min()/3), self[ith, ::sps].real.max() + np.abs(self[ith, ::sps].real.max()/3)])
            ax.set_ylim([self[ith, ::sps].imag.min() - np.abs(self[ith, ::sps].imag.min()/3), self[ith, ::sps].imag.max() + np.abs(self[ith, ::sps].imag.max()/3)])

        plt.tight_layout()


        plt.show()
        if flag:
            self.cuda()

    def inplace_normalise(self):
        factor = np.mean(np.abs(self[:])**2,axis=1,keepdims=True)
        self[:] = self[:]/np.sqrt(factor)

