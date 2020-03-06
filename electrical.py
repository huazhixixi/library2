class Resampler(object):

    @staticmethod
    def prop(signal, new_fs):

        if signal.is_on_cuda:
            import cusignal
            signal.samples = cusignal.resample_poly(signal[:], 1, signal.fs_in_fiber / new_fs, axis=-1)
        else:
            import resampy
            signal.samples = resampy.resample(signal[:], signal.fs_in_fiber, new_fs, axis=-1, filter='kaiser_fast')

        signal.sps = new_fs / signal.baudrate
        signal.sps = int(signal.sps)
        return signal


