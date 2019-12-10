from __future__ import division

import numpy as np
import pickle
import random
import librosa
import scipy


random.seed(4264523625)

import scipy
from scipy.signal import lfilter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import paired_distances


def dumpPickle(d, name, path):
    
    with open(path + name, 'wb') as output:
    # Pickle dictionary using protocol 0.
        pickle.dump(d, output)
    print('%s Saved' % (name))
    
def highpass(x, fo, sr, tabs = 10001):

    h = scipy.signal.firwin(tabs, fo, fs=sr, pass_zero=False)
    z = scipy.signal.lfilter(h, [1], x)
    return z

def lowpass(x, fo, sr, tabs = 10000):
    h = scipy.signal.firwin(tabs, fo, fs=sr, pass_zero=True)
    z = scipy.signal.lfilter(h, [1], x)
    return z

def Filter(x, sr, fType):
    audio = [] # x must be of tensor shape (note, time, 1)
    if fType == 'highpass':
        for a in x:
            audio.append(highpass(a[:,0], 800, sr, tabs = 10001))
    elif fType == 'lowpass':
        for a in x:
            audio.append(lowpass(a[:,0], 800, sr, tabs = 10000))
    return np.expand_dims(np.asarray(audio),-1)   
      


def slicing(x, win_length, hop_length, center = True, windowing = True, pad = 0):
    # Pad the time series so that frames are centered
    if center:
        x = np.pad(x, int((win_length+pad) // 2), mode='constant')
    # Window the time series.
    y_frames = librosa.util.frame(x, frame_length=win_length, hop_length=hop_length)
    if windowing:
        window = scipy.signal.hann(win_length, sym=False)
    else:
        window = 1.0 
    f = []
    for i in range(len(y_frames.T)):
        f.append(y_frames.T[i]*window)
    return np.float32(np.asarray(f)) 


def overlap(x, x_len, win_length, hop_length, windowing = True, rate = 1): 
    x = x.reshape(x.shape[0],x.shape[1]).T
    if windowing:
        window = scipy.signal.hann(win_length, sym=False)
        rate = rate*hop_length/win_length
    else:
        window = 1
        rate = 1
    n_frames = x_len / hop_length
    expected_signal_len = int(win_length + hop_length * (n_frames))
    y = np.zeros(expected_signal_len)
    for i in range(int(n_frames)):
            sample = i * hop_length 
            w = x[:, i]
            y[sample:(sample + win_length)] = y[sample:(sample + win_length)] + w*window
    y = y[int(win_length // 2):-int(win_length // 2)]
    return np.float32(y*rate)   


def cropAndPad(x, crop = 0, pad = None):
    X = []
    for x_ in x:
        X_ = (x_[crop:,0])
        if pad:
            zeros = np.zeros((pad,))
            X_ = np.concatenate((X_,zeros))
        X.append(X_)
    X = np.asarray(X)
    
    return X.reshape(x.shape[0],-1,1)   

# objective Metrics


def getDistances(x,y):

    distances = {}
    distances['mae'] = mean_absolute_error(x, y)
    distances['mse'] = mean_squared_error(x, y)
    distances['euclidean'] = np.mean(paired_distances(x, y, metric='euclidean'))
    distances['manhattan'] = np.mean(paired_distances(x, y, metric='manhattan'))
    distances['cosine'] = np.mean(paired_distances(x, y, metric='cosine'))
   
    distances['mae'] = round(distances['mae'], 5)
    distances['mse'] = round(distances['mse'], 5)
    distances['euclidean'] = round(distances['euclidean'], 5)
    distances['manhattan'] = round(distances['manhattan'], 5)
    distances['cosine'] = round(distances['cosine'], 5)
    
    return distances

# mae

def getMAEnormalized(ytrue, ypred):
    
    ratio = np.mean(np.abs(ytrue))/np.mean(np.abs(ypred))

    return mean_absolute_error(ytrue, ratio*ypred)

# mfcc_cosine


def getMFCC(x, sr, mels=40, mfcc=13, mean_norm=False):
    
    melspec = librosa.feature.melspectrogram(y=x, sr=sr, S=None,
                                     n_fft=4096, hop_length=2048,
                                     n_mels=mels, power=2.0)
    melspec_dB = librosa.power_to_db(melspec, ref=np.max)
    mfcc = librosa.feature.mfcc(S=melspec_dB, sr=sr, n_mfcc=mfcc)
    if mean_norm:
        mfcc -= (np.mean(mfcc, axis=0))
    return mfcc

        
def getMSE_MFCC(y_true, y_pred, sr, mels=40, mfcc=13, mean_norm=False):
    
    ratio = np.mean(np.abs(y_true))/np.mean(np.abs(y_pred))
    y_pred =  ratio*y_pred
    
    y_mfcc = getMFCC(y_true, sr, mels=mels, mfcc=mfcc, mean_norm=mean_norm)
    z_mfcc = getMFCC(y_pred, sr, mels=mels, mfcc=mfcc, mean_norm=mean_norm)
    
    return getDistances(y_mfcc[:,:], z_mfcc[:,:]) 

