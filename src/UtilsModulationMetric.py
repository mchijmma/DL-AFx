from __future__ import division
from __future__ import print_function

from brian import Hz
from brian.hears import erbspace

import brian
from brian.hears import erbspace, Gammatone, Sound
from brian import Hz
from scipy.signal import hilbert
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import paired_distances
from collections import OrderedDict
import sys
import os
# from fnmatch import fnmatch

# import librosa
# import librosa.display
import scipy
import numpy as np 
import pandas as pd
# from sklearn.preprocessing import normalize
# from sklearn.metrics import mean_absolute_error




kGfmin, kGfmax, kGbands = 26, 6950, 12
kMfmin, kMfmax, kMbands = 0.5, 100, 12
kEsr = 400
kSR = 16000
cfG = erbspace(kGfmin*Hz, kGfmax*Hz, kGbands)
cfM = erbspace(kMfmin*Hz, kMfmax*Hz, kMbands)

# kEfmin = [0.5, 4.5, 10.5, 20.5]
# kEfmax = [4. , 10., 20., 100. ]

kEfmin = cfM[:-1]
kEfmax = cfM[1:]


def getGammatone(x, fmin, fmax, bands, sr):
    cf = erbspace(fmin*Hz, fmax*Hz, bands)
    gfb = Gammatone(Sound(x, samplerate=sr*Hz), cf)
    gamma = gfb.process()
    return gamma

def getFFT(x):

    n = len(x) # length of the signal

    x = x*scipy.signal.hann(n, sym=False)
    XD = np.fft.fft(x)/n # fft computing and normalization
    XD = XD[range(n//2)]
    
    return XD

def getModulation(x, fmin, fmax, bands, sr):
    cf = erbspace(fmin*Hz, fmax*Hz, bands)
    m = []
    for i in range(x.shape[1]):
        gfb = Gammatone(Sound(x[:,i], samplerate=sr*Hz), cf)
        m.append(gfb.process())
    return np.asarray(m)

def getEnvelope(x, power = True, downsample = None):
    
    envs = []
    for i in range(x.shape[1]):
        analytic_signal = hilbert(x[:,i])
        amplitude_envelope = (np.abs(analytic_signal))
        if power:
            amplitude_envelope = np.power(amplitude_envelope, 0.3)
        if downsample:
#             amplitude_envelope = librosa.resample(amplitude_envelope, downsample[0], downsample[1]) 
            amplitude_envelope = scipy.signal.resample(amplitude_envelope, len(amplitude_envelope)//(downsample[0]//downsample[1]))
        envs.append(amplitude_envelope)
    envs = np.asarray(envs)
    
    return envs.T

def getMarginal(x):
    
    stats = OrderedDict()
    marginal = scipy.stats.describe(x)
    stats['mean'] = marginal[2]
    stats['var'] = marginal[3]/(marginal[2]**2)
    stats['skew'] = marginal[4]
    stats['kurtosis'] = marginal[5]
    return stats

def plotSubBands(X, sr):
    plt.figure(figsize=(18, 8))
    t = np.linspace(0, len(X[:,0])/sr, num=len(X[:,0]))
    for i in range(X.shape[1]):
        plf.plot(t, X[:,i])
        

def getModulationPower(x_m, x_e):
    m = []
    for i in range(kGbands):
        a = scipy.stats.describe(x_m[i]**2)
        m.append(a[2])#/x_e['var'][i])
#         m.append(np.mean(x_m[i])/x_e['var'][i])
    return np.asarray(m) 

def getModulationSpectrum(x):
    x_mD = []
    for j in range(x.shape[0]):
        ffts = []
        for i in range(x.shape[2]):
            XD = getFFT(x[j,:,i])
            ffts.append(abs(XD))
        x_mD.append(ffts)
    x_mD = np.asarray(x_mD)
    return x_mD

def getModulationSpectrumEnergy(x, f1, f2, sr, normalizeDC = True):
    
    #returns modulation spectrum and modulation energy within specifc frequency bands
    energy = np.mean(x, axis = 1)**2
    if normalizeDC:
        energy = np.mean(energy, axis = 0)/np.mean(energy, axis = 0)[0]
    else:
        energy = np.mean(energy, axis = 0)
    
    energyBands = []
    for fmin, fmax in zip(f1, f2):
        
        bin1 = int(np.round(energy.shape[0]*fmin/(sr/2)))
        bin2 = int(np.round(energy.shape[0]*fmax/(sr/2)))
#         print(bin1, bin2)
        energyBands.append(np.sum(energy[bin1:bin2+1]))
    
    modulationspectrum = np.mean(x, axis = 1)
    modulationspectrum = np.mean(modulationspectrum, axis = 0)
        
    return modulationspectrum, np.asarray(energyBands)

def getMeanLogModulationSpectrum(x):
    
    x = np.mean(x, axis=0)
    x = np.mean(x, axis=0)
    x = np.log(x + 1e-10)
#     x = np.expand_dims(x, axis=1)
    return x

def getMP(audio, kLen):

    x_modulationEnergyBands = []

   
    x_g = getGammatone(audio[:kLen], kGfmin, kGfmax, kGbands, kSR)
    x_ge = getEnvelope(x_g, downsample = ((kSR, kEsr)), power = False)
    x_gem = getModulation(x_ge, kMfmin, kMfmax, kMbands, kEsr)
    x_ge_stats = getMarginal(x_ge)
    m_x = getModulationPower(x_gem, x_ge_stats)
    x_gemD = getModulationSpectrum(x_gem)
    x_Em, x_Ebm = getModulationSpectrumEnergy(x_gemD, kEfmin, kEfmax, kEsr, normalizeDC = True)
    x_em_stats = getMarginal(x_Em[1:])
    x_gemD_meanlog = getMeanLogModulationSpectrum(x_gemD)
    
    
    return x_gemD_meanlog, x_Ebm, x_em_stats.values()