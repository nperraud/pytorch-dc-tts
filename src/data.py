from wavio import read
from scipy.signal import firwin
import numpy as np

def downsample(sig,Nwin=32):
    win = firwin(numtaps=Nwin, cutoff=0.55)
    new_sig = sig.copy()
    new_sig = np.convolve(new_sig,win, 'same')
    new_sig = new_sig[::2]
    return new_sig

def normalize(x):
    m = np.max(np.abs(x))
    return x/m*0.95

def toint16(x):
    return np.int16(x*(2**15))

def tofloat(x):
    return np.float64(x)/2**15

def tomono(x):
    return (x[:,0]+x[:,1])/2

def get_song(ds=1):
    res = read('../data/despacito.wav')
    sig = tomono(tofloat(res.data))
    fs  = res.rate
    for _ in range(np.int(np.log2(ds))):
        fs = fs//2
        sig = downsample(sig)
    return normalize(sig), fs