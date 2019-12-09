import soundfile as sf
import numpy as np
import scipy.signal
import scipy.ndimage
import random



def getSameLenghts(model_config, inputAudio, effectAudio):
    
    audio_shape = effectAudio.shape[0]
    x_shape = inputAudio.shape[0]
    pad = x_shape - audio_shape
    if pad < 0:
        effectAudio = effectAudio[:pad]
    elif pad > 0:
        effectAudio = effectAudio[:,:-pad]
        zeros = (model_config['lengthSamples']) - (audio_shape-2*model_config['samplingRate']+model_config['lengthSamples'])%(model_config['lengthSamples'])
        audio_1 = np.pad(audio_1, (0, zeros), 'constant', constant_values=(0, 0))
        inputAudio = np.pad(inputAudio, (0, zeros), 'constant', constant_values=(0, 0))
        
    return inputAudio, effectAudio

def getTimeAlligned(model_config, effectAudio):
    
    kSR = model_config['samplingRate']
    effectAudio = effectAudio[:]
    effectAudio_ = effectAudio[0:2*kSR]
    effectAudio_ = np.abs(effectAudio_)
    peaks, p = scipy.signal.find_peaks(effectAudio_, height=0.3)
    sample = peaks[np.argmax(p['peak_heights'])]
    time_shift = sample - kSR
    
    effectAudio = scipy.ndimage.interpolation.shift(effectAudio,-(time_shift),mode='nearest')
    
    return effectAudio

def getAsTensors(model_config, audio):
    
    kSR = model_config['samplingRate']
    audio = audio[2*kSR:]
    audio = np.reshape(audio,(-1,int(model_config['lengthSamples']),1))
    audio = audio.astype(np.float32)
    
    return audio

def getTrainingTensors(model_config, input_1, effect_1, input_2, effect_2):
    
    Xtrain = np.vstack((input_1, input_2))
    Ytrain = np.vstack((effect_1, effect_2))

    Xtrain = Xtrain.astype(np.float32)
    Ytrain = Ytrain.astype(np.float32)
       
    C1 = 0.*np.ones(input_1.shape[0])
    C2 = 1.*np.ones(input_2.shape[0])
    C3 = np.asarray([random.uniform(0,1) for _ in xrange(Xtrain.shape[0])])
    XtrainC = np.hstack((C1, C2, C3))  
    
    zeros = np.zeros_like(Xtrain)
    Xtrain = np.vstack((Xtrain, zeros))
    Ytrain = np.vstack((Ytrain, zeros))
    
    return Xtrain, Ytrain, XtrainC
    
    
    

def processAudioFiles(model_config, inputPath, effectPath1, effectPath2):
    
    X, _ = sf.read(inputPath)
    Y_1, _ = sf.read(effectPath1)
    Y_2, _ = sf.read(effectPath2)
    
    if len(Y_1.shape) > 1:
        Y_1 = Y_1[:,0]
    if len(Y_2.shape) > 1:
        Y_2 = Y_2[:,0]
    
    
    X_1, Y_1 = getSameLenghts(model_config, X, Y_1)
    X_2, Y_2 = getSameLenghts(model_config, X, Y_2)
    
    
    Y_1 = getTimeAlligned(model_config, Y_1)
    Y_2 = getTimeAlligned(model_config, Y_2)
    
    X_1 = getAsTensors(model_config, X_1)
    Y_1 = getAsTensors(model_config, Y_1)
    X_2 = getAsTensors(model_config, X_2)
    Y_2 = getAsTensors(model_config, Y_2)
    
    Xtrain, Ytrain, XtrainC = getTrainingTensors(model_config, X_1, Y_1, X_2, Y_2)
    
    return Xtrain, Ytrain, XtrainC
    
    