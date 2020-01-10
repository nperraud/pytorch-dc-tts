import os
import copy
import librosa
import scipy.io.wavfile
import numpy as np

from tqdm import tqdm
from scipy import signal

from stft4pghi.stft import GaussTF
from datasets.lj_speech import LJSpeech

from hparams import HParams as hp


# from src.spectrogramInverter import SpectrogramInverter


def load_signal(fpath, sr=22050):
    # Loading sound file
    y, sr = librosa.load(fpath, sr=sr)
    # Trimming
    y, _ = librosa.effects.trim(y)
    # Preemphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    y = y[:1024*(len(y)//1024)]
    return y


def make_spectrograms(y, a, M, n_mels, sr=22050):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      y  : sound file
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    
    tfsystem = GaussTF(a=a, M=M)
    linear = tfsystem.dgt(y)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)
    mag = mag/np.max(mag)
    
    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, M, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = np.log10(np.maximum(1e-5, mel))/2.5+1
    mag = np.log10(np.maximum(1e-5, mag))/2.5+1
    
#     # normalize
#     mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
#     mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.astype(np.float32)  # (T, n_mels)
    mag = mag.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag





def preprocess(dataset_path):
    """Preprocess the given dataset."""
    rr = hp.reduction_rate # 2
    a = hp.hop_length # 256
    M = hp.win_length # 1024
    n_mels = hp.n_mels # 80
    sr = hp.sr # sr=22050
    speech_dataset = LJSpeech([])

    wavs_path = os.path.join(dataset_path, 'wavs')
    mels_path = os.path.join(dataset_path, 'mels')
    if not os.path.isdir(mels_path):
        os.mkdir(mels_path)
    mags_path = os.path.join(dataset_path, 'mags')
    if not os.path.isdir(mags_path):
        os.mkdir(mags_path)

    for fname in tqdm(speech_dataset.fnames):
        y = load_signal(os.path.join(wavs_path, '%s.wav' % fname),sr)
        mel, mag = make_spectrograms(y, a, M, n_mels, sr=sr)
        
        # Reduction
#         mel = mel[::rr, :]
        tmp = np.zeros([mel.shape[0], mel.shape[1]//rr], np.float32)
        for i in range(rr):
            tmp += mel[:, i::rr]
        mel = tmp/rr
        
        # Saving the data: here I transpose it because text2mel works like this.
        # Also text2mel does from 0 to 1...
        # We can invert this later...
        np.save(os.path.join(mels_path, '%s.npy' % fname), mel.T/2.02+1)
        np.save(os.path.join(mags_path, '%s.npy' % fname), mag.T/2.02+1) 

        

# def spectrogram2wav(mag):
#     '''# Generate wave file from linear magnitude spectrogram
#     Args:
#       mag: A numpy array of (T, 1+n_fft//2)
#     Returns:
#       wav: A 1-D numpy array.
#     '''
#     # transpose
#     mag = mag.T

#     # de-noramlize
#     mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

#     # to amplitude
#     mag = np.power(10.0, mag* 0.05)

#     # wav reconstruction
#     wav = griffin_lim(mag ** hp.power)
# #     length = len(invert_spectrogram(mag))
# #     inverter = SpectrogramInverter(hp.win_length, hp.hop_length, length)
# #     wav = inverter._invertSpectrogram(mag[:-1]** hp.power)
#     # de-preemphasis
#     wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

#     # trim
#     wav, _ = librosa.effects.trim(wav)

#     return wav.astype(np.float32)


# def griffin_lim(spectrogram):
#     '''Applies Griffin-Lim's raw.'''
#     X_best = copy.deepcopy(spectrogram)
#     for i in range(hp.n_iter):
#         X_t = invert_spectrogram(X_best)
# #         est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
#         g_analysis = {'name': 'gauss', 'M': hp.win_length}
#         est = ltfatpy.dgtreal(X_t.astype(np.float64), g_analysis, hp.hop_length, hp.win_length)[0]
#         phase = est / np.maximum(1e-8, np.abs(est))
#         X_best = spectrogram * phase
#     X_t = invert_spectrogram(X_best)
#     y = np.real(X_t)

#     return y


# def invert_spectrogram(spectrogram):
#     '''Applies inverse fft.
#     Args:
#       spectrogram: [1+n_fft//2, t]
#     '''
# #     g_analysis = {'name': 'gauss', 'M': hp.win_length}
# #     g_synthesis = {'name': ('dual', g_analysis['name']), 'M': g_analysis['M']}
# #     return ltfatpy.idgtreal(spectrogram.astype(np.complex128), g_synthesis,  hp.hop_length, hp.win_length)[0]

#     tfsystem = GaussTF(a=hp.hop_length,M=hp.win_length)
# #     return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

# def save_to_wav(mag, filename):
#     """Generate and save an audio file from the given linear spectrogram using Griffin-Lim."""
#     wav = spectrogram2wav(mag)
#     scipy.io.wavfile.write(filename, hp.sr, wav)
