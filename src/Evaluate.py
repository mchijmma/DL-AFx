from __future__ import division
from __future__ import print_function

import sys
import os

from sacred import Experiment
from Config import config_ingredient
import Models
from Layers import Generator, GeneratorWaveNet, GeneratorContext
import Utils
import UtilsModulationMetric
import librosa
import numpy as np
import math
import json

from __main__ import kSR, kContext

def evaluate(cfg, model_type, nameModel):
   
    model_config = cfg[model_type]

    print('Evaluating ', model_config['modelName'], model_type)

    # Xtest, Ytestshould be tensors of shape (number_of_recordings, number_of_samples, 1) 
    Xtest = np.random.rand(1, 2*kSR, 1)
    Ytest = np.random.rand(1, 2*kSR, 1)
    
    # zero pad at the end as well. 
    Xtest = Utils.cropAndPad(Xtest, crop = 0, pad = kContext*model_config['winLength']//2)
    Ytest = Utils.cropAndPad(Ytest, crop = 0, pad = kContext*model_config['winLength']//2)
    
    kLen = Xtest.shape[1]
    kBatch = int((kLen/(model_config['winLength']//2)) + 1)

    if model_type in 'CAFx':

        model = Models.CAFx(model_config['winLength'],
                            model_config['filters'], 
                            model_config['kernelSize'], 
                            model_config['learningRate'])


        testGen = Generator(Xtest, Ytest, model_config['winLength'], model_config['winLength']//2)

    elif model_type in 'WaveNet':

        model = Models.WaveNet(model_config['learningRate'],
                               model_config['wavenetConfig'])


        testGen = GeneratorWaveNet(Xtest, Ytest, model_config['wavenetConfig']['model']['input_length'],
                                        model_config['winLength'], model_config['winLength']//2)


    elif model_type in 'CRAFx':

        model = Models.CRAFx(model_config['winLength'],
                            model_config['filters'], 
                            model_config['kernelSize'], 
                            model_config['learningRate'])

        testGen = GeneratorContext(Xtest, Ytest, kContext, model_config['winLength'], model_config['winLength']//2)


    elif model_type in 'CWAFx':

        model = Models.CWAFx(model_config['winLength'],
                            model_config['filters'], 
                            model_config['kernelSize'], 
                            model_config['learningRate'], 
                            model_config['wavenetConfig'])

        testGen = GeneratorContext(Xtest, Ytest, kContext, model_config['winLength'], model_config['winLength']//2)



    # load trained model
    
    model.load_weights(model_config['modelsPath']+nameModel+'_chk.h5', by_name=True) 
    
    if os.path.isdir('./Audio_'+nameModel) == False:
        os.mkdir('./Audio_'+nameModel)
   
    metrics = {}
    mae = []
    mfcc_cosine = []
    mse_y = []
    mse_z = []
    for idx in range(Xtest.shape[0]):

        x = testGen[idx][0]
        Z =  model.predict(x, batch_size=kBatch)
        Ztest_waveform = Utils.overlap(Z, kLen,
                                       model_config['winLength'], model_config['winLength']//2, windowing=True, rate=2)
        
        Ytest_waveform = Ytest[idx].reshape(Ytest[idx].shape[0])

        librosa.output.write_wav('./Audio_'+nameModel+'/'+nameModel+'_'+str(idx)+'.wav',
                             Ztest_waveform, kSR, norm=False)
        
        
        mae.append(Utils.getMAEnormalized(Ytest_waveform, Ztest_waveform))
        d = Utils.getMSE_MFCC(Ytest_waveform, Ztest_waveform, kSR, mean_norm=False)    
        mfcc_cosine.append(d['cosine'])
        ms, e_y, _ = UtilsModulationMetric.getMP(Ytest_waveform, kLen)
        ms, e_z, _ = UtilsModulationMetric.getMP(Ztest_waveform, kLen)
        mse_y.append(e_y)
        mse_z.append(e_z)
        
    d = Utils.getDistances(np.asarray(mse_y), np.asarray(mse_z))    
    metrics['mae'] = round(np.mean(mae), 5)
    metrics['mfcc_cosine'] = round(np.mean(mfcc_cosine), 5)
    metrics['msed'] = round(np.mean(d['euclidean']), 5)
    
    for metric in metrics.items():
            print(metric)
        
    with open(model_config['modelsPath']+nameModel+'_metrics.json', 'w') as outfile:
        json.dump(metrics, outfile)
        
 
    print('Evaluation finished.')
