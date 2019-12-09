import numpy as np
import pickle
import random
import librosa
import scipy

# from __main__ import model_config['seed']

random.seed(4264523625)

import scipy
from scipy.signal import lfilter


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
    
def trimDataset(x, pre = 0, post = None):
    X = []
    for x_ in x:
        X_ = (x_[pre:,0])
        if post:
            zeros = np.zeros((post,))
            X_ = np.concatenate((X_,zeros))
        X.append(X_)
    X = np.asarray(X)
    
    return X.reshape(x.shape[0],-1,1)      


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



    
# def compute_receptive_field_length(stacks, dilations, filter_length, target_field_length):

#     half_filter_length = (filter_length-1)/2
#     length = 0
#     for d in dilations:
#         length += d*half_filter_length
#     length = 2*length
#     length = stacks * length
#     length += target_field_length
#     return int(length)

