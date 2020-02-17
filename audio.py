import os
import copy
import librosa
import scipy.io.wavfile
import numpy as np

from tqdm import tqdm
from scipy import signal

from tifresi.pipelines.LJspeech import compute_mag_mel_from_path
from datasets.lj_speech import LJSpeech

from hparams import HParams as hp


# from src.spectrogramInverter import SpectrogramInverter





def preprocess(dataset_path):
    """Preprocess the given dataset."""
#     rr = hp.reduction_rate # 2
#     a = hp.hop_length # 256
#     M = hp.win_length # 1024
#     n_mels = hp.n_mels # 80
#     sr = hp.sr # sr=22050
    speech_dataset = LJSpeech([])

    wavs_path = os.path.join(dataset_path, 'wavs')
    mels_path = os.path.join(dataset_path, 'mels')
    if not os.path.isdir(mels_path):
        os.mkdir(mels_path)
    mags_path = os.path.join(dataset_path, 'mags')
    if not os.path.isdir(mags_path):
        os.mkdir(mags_path)

    for fname in tqdm(speech_dataset.fnames):
#         y = load_signal(os.path.join(wavs_path, '%s.wav' % fname),sr)
            
        mel, mag = compute_mag_mel_from_path(os.path.join(wavs_path, '%s.wav' % fname))
        
        # Saving the data: here I transpose it because text2mel works like this.
        # Also text2mel does from 0 to 1...
        # We can invert this later...
        mel = mel.T
        max_mel = np.max(mel)
        min_mel = np.min(mel)
        lim = 0.001
        if max_mel-min_mel>1-2*lim:
            mel = mel - max_mel + 1-lim
        else:
            mel = mel - min_mel + lim
        mel = np.clip(mel, lim, None )
        
#         mag = ((mag.T+1)/2.02)+0.01
        mag = mag.T
        assert(np.max(mel)<=1)
        assert(np.min(mel)>=0)
        assert(np.max(mag)<=1)
        assert(np.min(mag)>=0)
        np.save(os.path.join(mels_path, '%s.npy' % fname), mel)
        np.save(os.path.join(mags_path, '%s.npy' % fname), mag) 

        

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
