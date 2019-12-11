from __future__ import division
from __future__ import print_function

import sys
import os

from sacred import Experiment
from Config import config_ingredient
import Models
from Layers import Generator, GeneratorWaveNet, GeneratorContext
import Utils

ex = Experiment('model training', ingredients=[config_ingredient])

import Evaluate
import numpy as np
import math

# model_type should be 'pretraining', 'CAFx', 'WaveNet', 'CRAFx' or 'CWAFx'

@ex.automain
def main(cfg, model_type):
   
    model_config = cfg[model_type]
    
    if os.path.isdir(model_config['modelsPath']) == False:
        os.mkdir(model_config['modelsPath'])

    i = 0
    while os.path.exists(model_config['modelsPath']+'model_%s.pkl' % i):
        i += 1
    model_config['modelName'] = 'model_%s' % i

    Utils.dumpPickle(model_config, model_config['modelName']+ '.pkl', model_config['modelsPath'])
    
    try:

        print('Training ', model_config['modelName'])

        # Xtrain, Ytrain, Xval, Yval should be tensors of shape (number_of_recordings, number_of_samples, 1) 
        Xtrain = np.random.rand(1, 32000, 1)
        Ytrain = np.random.rand(1, 32000, 1)
        Xval = np.random.rand(1, 32000, 1)
        Yval = np.random.rand(1, 32000, 1)
        
        # since the samples are 2 secs long, we zero pad 4*hop_size samples at the end of the recording. This for the 4 
        # subsequent frames in the Leslie modeling tasks.
        Xtrain = Utils.cropAndPad(Xtrain, crop = 0, pad = 4*model_config['winLength']//2)
        Ytrain = Utils.cropAndPad(Ytrain, crop = 0, pad = 4*model_config['winLength']//2)
        Xval = Utils.cropAndPad(Xval, crop = 0, pad = 4*model_config['winLength']//2)
        Yval = Utils.cropAndPad(Yval, crop = 0, pad = 4*model_config['winLength']//2)


        if model_type in 'pretraining':

            model = Models.pretrainingModel(model_config['winLength'],
                                model_config['filters'], 
                                model_config['kernelSize'], 
                                model_config['learningRate'])


            trainGen = Generator(Xtrain, Ytrain, model_config['winLength'], model_config['winLength']//2)
            valGen = Generator(Xval, Yval, model_config['winLength'], model_config['winLength']//2)

        elif model_type in 'CAFx':

            model = Models.CAFx(model_config['winLength'],
                                model_config['filters'], 
                                model_config['kernelSize'], 
                                model_config['learningRate'])


            trainGen = Generator(Xtrain, Ytrain, model_config['winLength'], model_config['winLength']//2)
            valGen = Generator(Xval, Yval, model_config['winLength'], model_config['winLength']//2)

        elif model_type in 'WaveNet':

            model = Models.WaveNet(model_config['learningRate'],
                                   model_config['wavenetConfig'])


            trainGen = GeneratorWaveNet(Xtrain, Ytrain, model_config['wavenetConfig']['model']['input_length'],
                                        model_config['winLength'], model_config['winLength']//2)
            valGen = GeneratorWaveNet(Xval, Yval, model_config['wavenetConfig']['model']['input_length'],
                                        model_config['winLength'], model_config['winLength']//2)

        elif model_type in 'CRAFx':

            model = Models.CRAFx(model_config['winLength'],
                                model_config['filters'], 
                                model_config['kernelSize'], 
                                model_config['learningRate'])

            trainGen = GeneratorContext(Xtrain, Ytrain, 4, model_config['winLength'], model_config['winLength']//2)
            valGen = GeneratorContext(Xval, Yval, 4, model_config['winLength'], model_config['winLength']//2)

        elif model_type in 'CWAFx':

            model = Models.CWAFx(model_config['winLength'],
                                model_config['filters'], 
                                model_config['kernelSize'], 
                                model_config['learningRate'], 
                                model_config['wavenetConfig'])

            trainGen = GeneratorContext(Xtrain, Ytrain, 4, model_config['winLength'], model_config['winLength']//2)
            valGen = GeneratorContext(Xval, Yval, 4, model_config['winLength'], model_config['winLength']//2)

        model.summary()

        # load pretrained model if available:
    #         model.load_weights(path_preatrained_model, by_name=True) 



        earlyStopping = Models.EarlyStopping(monitor=model_config['monitorLoss'],
                                          min_delta=0,
                                          patience=25,
                                          verbose=1,
                                          mode='auto',
                                          baseline=None, restore_best_weights=False)

        checkpointer = Models.ModelCheckpoint(filepath=model_config['modelsPath']+model_config['modelName']+'_chk.h5',
                                           monitor=model_config['monitorLoss'],
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True)

        model.fit_generator(trainGen,
                           steps_per_epoch=None,
                           epochs=model_config['epoch'],
                           verbose=2,
                           callbacks = [checkpointer, earlyStopping],
                           validation_data = valGen,
                           validation_steps=len(Xval),
                           shuffle=True)

        print('Reducing Learning rate by 4')

        symbolic_weights = getattr(model.optimizer, 'weights')
        weight_values = Models.K.batch_get_value(symbolic_weights)

        model.compile(loss='mae',
                      optimizer=Models.Adam(lr=model_config['learningRate']/4))

        model.load_weights(model_config['modelsPath']+model_config['modelName']+'_chk.h5', by_name=True)

        model.fit_generator(trainGen,
                           steps_per_epoch=None,
                           epochs=model_config['epoch'],
                           verbose=2,
                           callbacks = [checkpointer, earlyStopping],
                           validation_data = valGen,
                           validation_steps=len(Xval),
                           shuffle=True)

        print('Training finished.')
        
        Evaluate.evaluate(cfg, model_type, model_config['modelName'])

    except Exception as e: 
        print(e)
        print('Training failed: ' + model_config['modelName'] + ' was removed')  
        os.remove(model_config['modelsPath']+model_config['modelName']+'.pkl')
    

    
    


   