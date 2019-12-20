from librosa.filters import mel
import numpy as np
import torch

__author__ = 'Andres'


class MelspectrogramProcessor(object):
	def __init__(self, sampling_rate, fft_size, batch_size, mel_count, fmin, fmax):
		super().__init__()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		mel_basis = mel(sampling_rate, fft_size, mel_count, fmin=fmin, fmax=fmax)
		mel_pseudo_inverse = np.linalg.pinv(mel_basis)
		mel_pseudo_inverse[mel_basis.T == 0] = 0
		mel_basis = np.reshape(mel_basis, (1, 1, *mel_basis.shape))
		mel_pseudo_inverse = np.reshape(mel_pseudo_inverse, (1, 1, *mel_pseudo_inverse.shape))
		self.mel_basis = torch.from_numpy(np.repeat(mel_basis, batch_size, axis=0)).to(device)
		self.mel_pseudo_inverse = torch.from_numpy(np.repeat(mel_pseudo_inverse, batch_size, axis=0)).to(device)

	def computeMelOnLogMagSpectrograms(self, logMagSpectrograms):
		return torch.matmul(self.mel_basis[:, :, :, :256], torch.exp(5*(logMagSpectrograms-1)))

	def invertMelSpectrogram(self, melSpectrogram):
		unmelled_spectrograms = torch.matmul(self.mel_pseudo_inverse[:, :, :256, :], melSpectrogram)
		unmelled_spectrograms = torch.clamp(unmelled_spectrograms, np.exp(-10), 1)
		unmelled_spectrograms = 1 + torch.log(unmelled_spectrograms) / 5
		return unmelled_spectrograms