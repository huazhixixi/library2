import numpy as np
from scipy.fft import fft, ifft, fftfreq


from scipy.constants import c

from .signal_define import QamSignal



class AwgnChannel(object):

    def __init__(self, snr):
        self.snr = snr
        self.snr_linear = 10 ** (self.snr / 10)

    def prop(self, signal, power,divided_factor = 1,is_dp = True):
        '''
            signal: The signal to be propagated
            power: in dbm
        '''
        if signal.is_on_cuda:
            import cupy as np
        else:
            import numpy as np

        power = 10**(power/10) /1000

        noise_power = power / self.snr_linear * signal.sps_in_fiber
        noise_power_xpol = noise_power / (int(is_dp)+1)/divided_factor

        seq = np.sqrt((noise_power_xpol / 2)) * (np.random.randn(int(is_dp)+1, len(signal)) + 1j * np.random.randn(int(is_dp)+1, len(signal)))

        signal[:] = signal[:] + seq
        return signal


class Fiber(object):

    def __init__(self, alpha, D, length, reference_wavelength,slope,accuracy,name='SSMF',**kwargs):
        '''
            :param alpha:db/km
            :D:s^2/km
            :length:km
            :reference_wavelength:nm

        '''
        self.alpha = alpha
        self.D = D
        self.length = length
        self.reference_wavelength = reference_wavelength  # nm
        self.fft = None
        self.ifft = None
        self.plan = None
        self.slope = slope
        self.accuracy = accuracy
        self.name = name

    def prop(self, signal):
        raise NotImplementedError

    @property
    def alphalin(self):
        alphalin = self.alpha / (10 * np.log10(np.exp(1)))
        return alphalin

    @property
    def beta2_reference(self):
        return -self.D * (self.reference_wavelength * 1e-12) ** 2 / 2 / np.pi / c / 1e-3

    def beta2(self, wave_length):
        '''

        :param wave_length: [m]
        :return: beta2 at wave_length [s^2/km]
        '''
        dw = 2 * np.pi * c * (1 / wave_length - 1 / (self.reference_wavelength * 1e-9))
        return self.beta2_reference + self.beta3_reference * dw

    @property
    def beta3_reference(self):
        res = (self.reference_wavelength * 1e-12 / 2 / np.pi / c / 1e-3) ** 2 * (
                2 * self.reference_wavelength * 1e-12 * self.D + (
                self.reference_wavelength * 1e-12) ** 2 * self.slope * 1e12)

        return res

    def leff(self, length):
        '''

        :param length: the length of a fiber [km]
        :return: the effective length [km]
        '''
        effective_length = 1 - np.exp(-self.alphalin * length)
        effective_length = effective_length / self.alphalin
        return effective_length



class NonlinearFiber(Fiber):

    def __init__(self, alpha,D,length,reference_wavelength,slope,accuracy, **kwargs):
        '''
            :param: kwargs:
                key: step_length
                key:gamma
        '''
        super(NonlinearFiber, self).__init__(alpha=alpha,D=D,length=length,reference_wavelength=reference_wavelength,slope = slope,accuracy = accuracy,**kwargs)
        self.step_length = kwargs.get('step_length', 20 / 1000)
        self.gamma = kwargs.get('gamma', 1.3)
        self.linear_prop = None

        self.__init_backend()


    def __init_backend(self):
        try:
            from cupyx.scipy.fft import fft
            from cupyx.scipy.fft import ifft
            from cupyx.scipy.fft import get_fft_plan
            from cupyx.scipy.fft import fftfreq
            import cupy as np
            self.fft = fft
            self.ifft = ifft
            self.plan = get_fft_plan
            self.np = np
            self.linear_prop = self.linear_prop_cupy_scipy
            self.fftfreq = fftfreq
        except  ImportError:

            try:
                import arrayfire as af
                self.fft = af.dft
                self.ifft = af.idft
                self.linear_prop = self.linear_prop_af
                self.np = af
                raise ImportError
                
            except (ImportError,RuntimeError) :
                from scipy.fft import fft,ifft,fftfreq
                self.fft = fft
                self.ifft = ifft
                self.fftfreq = fftfreq
                self.linear_prop = self.linear_prop_cupy_scipy
                class Plan:
                    def __enter__(self):
                        pass
                    def __exit__(self, exc_type, exc_val, exc_tb):
                        pass

                self.plan = Plan()
                import numpy as np
                self.np = np

        assert self.fft is not None
        assert self.ifft is not None
        assert self.linear_prop is not None
        if self.linear_prop is self.linear_prop_cupy_scipy:
            assert self.plan is not None

    @property
    def step_length_eff(self):
        return (1 - self.np.exp(-self.alphalin * self.step_length)) / self.alphalin

    def prop(self, signal):
        signal.cuda()
        nstep = self.length / self.step_length
        nstep = int(np.floor(nstep))
        
        freq = self.fftfreq(signal.shape[1], 1 / signal.fs_in_fiber)

        if self.accuracy.lower()=='single':
            signal.to_32complex()
            freq = self.np.array(freq,dtype=self.np.complex64)

            omeg = 2 * self.np.pi * freq
            D = -1j / 2 * self.beta2(signal.wavelength) * omeg ** 2
            D = self.np.array(D,dtype=np.complex64)

        N = 8 / 9 * 1j * self.gamma
        atten = -self.alphalin / 2
        last_step = self.length - self.step_length * nstep

        signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], self.step_length / 2)
        signal[0], signal[1] = self.nonlinear_prop(N, signal[0], signal[1])
        signal[0] = signal[0] * self.np.exp(atten * self.step_length / 2)
        signal[1] = signal[1] * self.np.exp(atten * self.step_length / 2)

        for _ in range(nstep - 1):
            signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], self.step_length)

            signal[0], signal[1] = self.nonlinear_prop(N, signal[0], signal[1])
            signal[0] = signal[0] * self.np.exp(atten * self.step_length)
            signal[1] = signal[1] * self.np.exp(atten * self.step_length)

        signal[0] = signal[0] * self.np.exp(atten * self.step_length / 2)
        signal[1] = signal[1] * self.np.exp(atten * self.step_length / 2)
        signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], self.step_length / 2)

        if last_step:
            last_step_eff = (1 - self.np.exp(-self.alphalin * last_step)) / self.alphalin
            signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], last_step / 2)
            signal[0], signal[1] = self.nonlinear_prop(N, signal[0], signal[1], last_step_eff)
            signal[0] = signal[0] * self.np.exp(atten * last_step)
            signal[1] = signal[1] * self.np.exp(atten * last_step)
            signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], last_step / 2)

        return signal

    def nonlinear_prop(self, N, time_x, time_y, step_length=None):
        if step_length is None:
            time_x = time_x * self.np.exp(
                N * self.step_length_eff * (self.np.abs(time_x) ** 2 + self.np.abs(
                    time_y) ** 2))
            time_y = time_y * self.np.exp(
                N * self.step_length_eff * (self.np.abs(time_x) ** 2 + self.np.abs(time_y) ** 2))
        else:
            time_x = time_x * self.np.exp(
                N * step_length * (self.np.abs(time_x) ** 2 + self.np.abs(
                    time_y) ** 2))
            time_y = time_y * self.np.exp(
                N * step_length * (self.np.abs(time_x) ** 2 + self.np.abs(time_y) ** 2))

        return time_x, time_y

    def linear_prop_cupy_scipy(self, D, timex, timey, length):
        if callable (self.plan) :
            self.plan = self.plan(timex,shape = timex.shape,axes=-1)
        with self.plan:
            freq_x = self.fft(timex, overwrite_x=True)
            freq_y = self.fft(timey, overwrite_x=True)

            freq_x = freq_x * self.np.exp(D * length)
            freq_y = freq_y * self.np.exp(D * length)

            time_x = self.ifft(freq_x, overwrite_x=True)
            time_y = self.ifft(freq_y, overwrite_x=True)
            return time_x, time_y

    def linear_prop_af(self,signal):
        raise NotImplementedError
class NonlinearFiberNew(Fiber):

    def __init__(self, signal_bandwidth, alpha=0.2, D=16.7, length=80,
                 reference_wavelength=1550, slope=0, accuracy='single', gamma=1.3, fwm_limitation=4):
        super().__init__(alpha, D, length, reference_wavelength, slope, accuracy)
        self.gamma = gamma
        self.fwm_limitation = fwm_limitation
        self.bw = signal_bandwidth

        self.transimitted_length = 0

    def init(self, signal):
        try:
            from cupyx.scipy.fft import fft
            from cupyx.scipy.fft import ifft
            from cupyx.scipy.fft import get_fft_plan
            from cupyx.scipy.fft import fftfreq
            import cupy as np
            self.fft = fft
            self.ifft = ifft
            self.plan = get_fft_plan
            self.fftfreq = fftfreq
            self.np = np
        except ImportError:
            import numpy as np
            from scipy.fft import fft, ifft, fftfreq
            self.np = np
            self.fft = fft
            self.ifft = ifft
            self.fftfreq = fftfreq

            class Plan:
                def __enter__(self):
                    pass

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

            self.plan = Plan()

        freq = self.fftfreq(signal.shape[1], 1 / signal.fs_in_fiber)
        omeg = 2 * self.np.pi * freq
        self.linear_op = -1j / 2 * self.beta2(signal.wavelength) * omeg ** 2
        self.N = 8 / 9 * 1j * self.gamma
        self.step = self.fwm_limitation / self.np.abs(self.beta2(signal.center_wavelength)) / ((
                2 * self.np.pi * self.bw) ** 2)
        if self.accuracy.lower() == 'single':
            signal.to_32complex()

            self.linear_op = self.np.array(self.linear_op, dtype=self.np.complex64)
            self.N = np.array(self.N,dtype=self.np.complex64)
            self.step = self.np.array(self.step,dtype=self.np.float32)

        self.peak_power = self.np.max(self.np.abs(signal[0]) ** 2 + self.np.abs(signal[1]) ** 2)

    def update_step(self):
        step_eff_next = self.np.exp(self.alphalin*self.step)*self.leff(self.step)
        step_next = self.np.log(1 - step_eff_next * self.alphalin) / (-self.alphalin)

        if self.transimitted_length + step_next > self.length or step_next > self.length or np.isnan(step_next):
            self.step = self.length - self.transimitted_length
        else:
            self.step = self.np.log(1 - step_eff_next * self.alphalin) / (-self.alphalin)

    def prop(self, signal):
        signal.cuda()
        self.init(signal)

        atten = -self.alphalin / 2

        while self.transimitted_length < self.length:
            signal = self.linear_prop(signal)
            signal = self.nonlinear_prop(signal)

            signal[0] = signal[0] * self.np.exp(atten * self.step / 2)
            signal[1] = signal[1] * self.np.exp(atten * self.step / 2)

            signal[0], signal[1] = self.linear_prop(signal)
            signal[0] = signal[0] * self.np.exp(atten * self.step / 2)
            signal[1] = signal[1] * self.np.exp(atten * self.step / 2)
            self.transimitted_length+=self.step
            self.update_step()

        return signal

    def linear_prop(self, signal):
        timex = signal[0]
        timey = signal[1]

        if callable(self.plan):
            self.plan = self.plan(timex, shape=timex.shape, axes=-1)
        with self.plan:
            freq_x = self.fft(timex, overwrite_x=True)
            freq_y = self.fft(timey, overwrite_x=True)

            freq_x = freq_x * self.np.exp(self.linear_op * self.step/2)
            freq_y = freq_y * self.np.exp(self.linear_op * self.step/2)

            time_x = self.ifft(freq_x, overwrite_x=True)
            time_y = self.ifft(freq_y, overwrite_x=True)

            signal[0] = time_x
            signal[1] = time_y
        return signal

    def nonlinear_prop(self, signal):
        step_eff = self.leff(self.step)
        time_x = signal[0]
        time_y = signal[1]
        time_x = time_x * self.np.exp(
            self.N * step_eff * (self.np.abs(time_x) ** 2 + self.np.abs(
                time_y) ** 2))
        time_y = time_y * self.np.exp(
            self.N * step_eff * (self.np.abs(time_x) ** 2 + self.np.abs(time_y) ** 2))

        signal[0] = time_x
        signal[1] = time_y
        return signal

class LinearFiber(Fiber):
    '''
        property:
            self.alpha  [db/km]
            self.D [ps/nm/km]
            self.length [km]
            self.reference_wave_length:[nm]
            self.beta2: caculate beta2 from D,s^2/km
            self.slope: derivative of self.D ps/nm^2/km
            self.beta3_reference: s^3/km
        method:
            __call__: the signal will
    '''

    def __init__(self, alpha, D, length, slope=0, reference_wavelength=1550):
        super(LinearFiber, self).__init__(alpha,D,length,reference_wavelength,slope,None)


    def prop(self, signal:QamSignal):
        '''
        :param signal: signal object to propagation across this fiber
        :return: ndarray
        '''
        center_lambda = signal.center_wavelength

        after_prop = np.zeros_like(signal[:])
        for pol in range(0, signal.pol_number):
            sample = signal[pol, :]
            sample_fft = fft(sample)
            freq = fftfreq(signal[:].shape[1], 1 / signal.fs_in_fiber)
            omeg = 2 * np.pi * freq

            after_prop[pol, :] = sample_fft * np.exp(-self.alphalin * self.length / 2)
            after_prop[pol, :] = ifft(after_prop[pol, :])

            disp = np.exp(-1j / 2 * self.beta2(center_lambda) * omeg ** 2 * self.length)
            after_prop[pol, :] = ifft(fft(after_prop[pol, :]) * disp)

        signal.samples = after_prop
        return signal
