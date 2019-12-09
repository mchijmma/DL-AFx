from __future__ import division
from __future__ import print_function

import sys
import os

from sacred import Experiment
from Config import config_ingredient
import Models
from Layers import Generator, GeneratorWaveNet, GeneratorContext
import Utils
import librosa
import numpy as np
import math


def evaluate(cfg, model_type):
   
    model_config = cfg[model_type]

    print('Evaluating ', model_config['modelName'])

    # Xtest, Ytestshould be tensors of shape (number_of_recordings, number_of_samples, 1) 
    Xtest = np.random.rand(1, 32000, 1)
    Ytest = np.random.rand(1, 32000, 1)

    if model_type in 'pretraining':

        model = Models.pretrainingModel(model_config['winLength'],
                            model_config['filters'], 
                            model_config['kernelSize'], 
                            model_config['learningRate'])


        testGen = Generator(Xtest, Ytest, model_config['winLength'], model_config['winLength']//2)

    elif model_type in 'CAFx':

        model = Models.CAFx(model_config['winLength'],
                            model_config['filters'], 
                            model_config['kernelSize'], 
                            model_config['learningRate'])


        testGen = Generator(Xtest, Ytest, model_config['winLength'], model_config['winLength']//2)

    elif model_type in 'WaveNet':

        model = Models.WaveNet(model_config['learningRate'],
                               model_config['wavenetConfig'])


        testGen = GeneratorWaveNet(Xtest, Ytest, model_config['winLength'], 4096, model_config['winLength']//2)


    elif model_type in 'CRAFx':

        model = Models.CRAFx(model_config['winLength'],
                            model_config['filters'], 
                            model_config['kernelSize'], 
                            model_config['learningRate'])

        testGen = GeneratorContext(Xtest, Ytest, 4, model_config['winLength'], model_config['winLength']//2)


    elif model_type in 'CWAFx':

        model = Models.CWAFx(model_config['winLength'],
                            model_config['filters'], 
                            model_config['kernelSize'], 
                            model_config['learningRate'], 
                            model_config['wavenetConfig'])

        testGen = GeneratorContext(Xtest, Ytest, 4, model_config['winLength'], model_config['winLength']//2)



    # load trained model, in this case 'model_0_chk.h5':
    nameModel = 'model_0'
    model.load_weights(model_config['modelsPath']+nameModel+'_chk.h5', by_name=True) 
    
    os.mkdir('./Audio_'+nameModel)
   

    for idx in range(Xtest.shape[0]):

        x = testGen[idx][0]
        Z =  model.predict(x)

        Ztest_waveform = Utils.overlap(Z, 32000,
                                       model_config['winLength'], model_config['winLength']//2, windowing=True, rate=2)

        librosa.output.write_wav('./Audio_'+nameModel+'/'+nameModel+_+str(idx)+'.wav',
                             Ztest_waveform, 16000, norm=False)

    print('Evaluation finished.')
