import ltfatpy
import numpy as np

__author__ = 'Andres'


class LTFATStft(object):
    def __init__(self, windowLength, hopSize):
        super().__init__()
        self.windowLength = windowLength
        self.hopSize = hopSize
        self.g_analysis = {'name': 'gauss', 'M': windowLength}
        self.g_synthesis = {'name': ('dual', self.g_analysis['name']), 'M': self.g_analysis['M']}

    def oneSidedStft(self, signal):
        return ltfatpy.dgtreal(signal, self.g_analysis, self.hopSize, self.windowLength)[0]

    def inverseOneSidedStft(self, signal):
        return ltfatpy.idgtreal(signal, self.g_synthesis, self.hopSize, self.windowLength)[0]

    def magAndPhaseOneSidedStft(self, signal):
        stft = self.oneSidedStft(signal)
        return np.abs(stft), np.angle(stft)

    def log10MagAndPhaseOneSidedStft(self, signal, clipBelow=1e-14):
        realDGT = self.oneSidedStft(signal)
        return self.log10MagFromRealDGT(realDGT, clipBelow), np.angle(realDGT)

    def log10MagFromRealDGT(self, realDGT, clipBelow=1e-14):
        return np.log10(np.clip(np.abs(realDGT), a_min=clipBelow, a_max=None))

    def reconstructSignalFromLogged10Spectogram(self, logSpectrogram, phase):
        reComplexStft = (10 ** logSpectrogram) * np.exp(1.0j * phase)
        return self.inverseOneSidedStft(reComplexStft)

    def logMagAndPhaseOneSidedStft(self, signal, clipBelow=np.e**-30, normalize=False):
        realDGT = self.oneSidedStft(signal)
        spectrogram = self.logMagFromRealDGT(realDGT, clipBelow, normalize)
        return spectrogram, np.angle(realDGT)

    def logMagFromRealDGT(self, realDGT, clipBelow=np.e**-30, normalize=False):
        spectrogram = np.abs(realDGT)
        if normalize:
            spectrogram = spectrogram/np.max(spectrogram)
        return np.log(np.clip(spectrogram, a_min=clipBelow, a_max=None))

    def reconstructSignalFromLoggedSpectogram(self, logSpectrogram, phase):
        reComplexStft = (np.e ** logSpectrogram) * np.exp(1.0j * phase)
        return self.inverseOneSidedStft(reComplexStft)
