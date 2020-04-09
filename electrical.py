class Resampler(object):

    @staticmethod
    def prop(signal, new_sps):

        if signal.is_on_cuda:
            import cusignal
            signal.samples = cusignal.resample_poly(signal[:], signal.sps, new_sps, axis=-1)
        else:
            import resampy
            signal.samples = resampy.resample(signal[:], signal.sps, new_sps, axis=-1, filter='kaiser_fast')

        signal.sps = new_sps
        return signal


