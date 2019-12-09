from __future__ import division

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
import keras
from keras import backend as K
K.set_session(sess)

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Concatenate, TimeDistributed, Lambda, Reshape
from keras.layers import Multiply, Add, UpSampling1D, MaxPooling1D, Bidirectional, LSTM, GlobalAvgPool1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from Layers import Conv1D_local, Dense_local, SAAF, Conv1D_tied, Slice

import numpy as np

# model to train when initializing Conv1d and Conv1D-Local layers.

def BooleanMask(x):
  
    output = K.greater_equal(x[0], x[1])
    output = K.cast(output, dtype='float32')
    output = K.tf.multiply(output, x[1])

    return output

def toPermuteDimensions(x):
    
    return K.permute_dimensions(x, (0, 2, 1))

def pretrainingModel(win_length, filters, kernel_size_1, learning_rate):

    x = Input(shape=(win_length, 1), name='input')
    conv = Conv1D(filters, kernel_size_1, strides=1, padding='same',
                       kernel_initializer='lecun_uniform',
                       input_shape=(win_length, 1), name='conv')

    conv_smoothing = Conv1D_local(filters, kernel_size_1*2, strides=1, padding='same',
                                  kernel_initializer='lecun_uniform', name='conv_smoothing')

    deconv = Conv1D_tied(1, kernel_size_1, conv, padding='same', name='deconv')
    
     
    X = conv(x)
    X_abs = Activation(K.abs, name='conv_activation')(X)
    M = conv_smoothing(X_abs)
    M = Activation('softplus', name='conv_smoothing_activation')(M)
    P = X
    Z = MaxPooling1D(pool_size=win_length//64, name='max_pooling')(M)
    M_ = UpSampling1D(size=win_length//64, name='up_sampling_naive')(Z)   
    M_ = Lambda((BooleanMask), name='boolean_mask')([M,M_])
        
    Y = Multiply(name='phase_unpool_multiplication')([P,M_])
    Y = deconv(Y)
    
    model = Model(inputs=[x], outputs=[Y])
    
    model.compile(loss={'deconv': 'mae'},
                         loss_weights={'deconv': 1.0},
                        optimizer=Adam(lr=learning_rate))

    return model


#CAFx model. 

def CAFx(win_length, filters, kernel_size_1, learning_rate):

    x = Input(shape=(win_length, 1), name='input')

    conv = Conv1D(filters, kernel_size_1, strides=1, padding='same',
                       kernel_initializer='lecun_uniform',
                       input_shape=(win_length, 1), name='conv')
    
    conv_smoothing = Conv1D_local(filters, kernel_size_1*2, strides=1,
                                         padding='same', name='conv_smoothing',
                                         kernel_initializer='lecun_uniform')


    dense_in = Dense_local(win_length//64, activation='softplus', name='dense_local_in')

    deconv = Conv1D_tied(1, kernel_size_1, conv, padding='same', name='deconv')
    
    
    X = conv(x)
    X_abs = Activation(K.abs, name='conv_activation')(X)
    M = conv_smoothing(X_abs)
    M = Activation('softplus', name='conv_smoothing_activation')(M)
    P = X
    Z = MaxPooling1D(pool_size=win_length//64, name='max_pooling')(M)
      
    Z = Lambda((toPermuteDimensions), name='permute_dimensions_dnn_in')(Z)
    Z = dense_in(Z)
    Z = TimeDistributed(Dense(win_length//64, activation = 'softplus'), name='dense_out')(Z)
    Z = Lambda((toPermuteDimensions), name='permute_dimensions_dnn_out')(Z)
    
    M_ = UpSampling1D(size=win_length//64, name='up_sampling_naive')(Z)
    Y_ = Multiply(name='phase_unpool_multiplication')([P,M_])
    
    Y_ = Dense(filters, activation = 'relu', name = 'dense_saaf_in')(Y_)
    Y_ = Dense(filters//2, activation = 'relu', name = 'dense_saaf_h1')(Y_)
    Y_ = Dense(filters//2, activation = 'relu', name = 'dense_saaf_h2')(Y_)
    Y_ = Dense(filters//2, activation = 'relu', name = 'dense_saaf_h3')(Y_)
    Y_ = Dense(filters, activation = 'linear', name = 'dense_saaf_out')(Y_)
    
    Y_ = SAAF(break_points=25, break_range=0.2, magnitude=100, order=2, tied_feamap=True,
            kernel_initializer = 'random_normal', name = 'saaf_out')(Y_)
    
    Y = deconv(Y_)
    
    model = Model(inputs=[x], outputs=[Y])
    
    model.compile(loss={'deconv': 'mae'},
                        loss_weights={'deconv': 1.0},
                        optimizer=Adam(lr=learning_rate))
    


    return model


# CRAFx model.

def se_block(x, num_features, weight_decay=0., amplifying_ratio=16, idx = 1):
    x = Multiply(name='dnn-saaf-se_%s'%idx)([x, se_fn(x, amplifying_ratio, idx)])
    return x
def se_fn(x, amplifying_ratio, idx):
    num_features = x.shape[-1].value
    x = Activation(K.abs)(x)
    x = GlobalAvgPool1D()(x)
    x = Reshape((1, num_features))(x)
    x = Dense(num_features * amplifying_ratio, activation='relu', kernel_initializer='glorot_uniform',
              name='se_dense1_%s'%idx)(x)
    x = Dense(num_features, activation='sigmoid', kernel_initializer='glorot_uniform',
              name='se_dense2_%s'%idx)(x)
    return x




def CRAFx(win_length, filters, kernel_size_1, learning_rate):
   
    kContext = 4 # past and subsequent frames
    
    x = Input(shape=(kContext*2+1, win_length, 1), name='input')

    
    conv = Conv1D(filters, kernel_size_1, strides=1, padding='same',
                       kernel_initializer='lecun_uniform', input_shape=(win_length, 1))
    
    activation_abs = Activation(K.abs)
    activation_sp = Activation('softplus')
    max_pooling = MaxPooling1D(pool_size=win_length//64)

    conv_smoothing = Conv1D_local(filters, kernel_size_1*2, strides=1, padding='same',
                                  kernel_initializer='lecun_uniform')

    bi_rnn = Bidirectional(LSTM(filters*2, activation='tanh', stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1), merge_mode='concat', name='birnn_in')
    bi_rnn1 = Bidirectional(LSTM(filters, activation='tanh', stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1), merge_mode='concat', name='birnn_1')
    bi_rnn2 = Bidirectional(LSTM(filters//2, activation='linear', stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1), merge_mode='concat', name='birnn_2')
    
    deconv = Conv1D_tied(1, kernel_size_1, conv, padding='same', name='deconv')
    
    
    X = TimeDistributed(conv, name='conv')(x)
    X_abs = TimeDistributed(activation_abs, name='conv_activation')(X)
    M = TimeDistributed(conv_smoothing, name='conv_smoothing')(X_abs)
    M = TimeDistributed(activation_sp, name='conv_smoothing_activation')(M)
    P = X
    Z = TimeDistributed(max_pooling, name='max_pooling')(M)
    Z = Lambda(lambda inputs: tf.unstack(inputs, num=kContext*2+1, axis=1, name='unstack2'))(Z)
    Z = Concatenate(name='concatenate')(Z)

    Z = bi_rnn(Z)
    Z = bi_rnn1(Z)
    Z = bi_rnn2(Z)
    Z = SAAF(break_points=25, break_range=0.2, magnitude=100, order=2, tied_feamap=True,
            kernel_initializer = 'random_normal', name = 'saaf_1')(Z)

    M_ = UpSampling1D(size=win_length//64, name='up_sampling_naive')(Z)
    P = Lambda(lambda inputs: tf.unstack(inputs, num=kContext*2+1, axis=1, name='unstack'))(P)
    Y = Multiply(name='phase_unpool_multiplication')([P[kContext],M_])
  
    Y_ = Dense(filters, activation = 'tanh', name = 'dense_in')(Y)  
    Y_ = Dense(filters//2, activation = 'tanh', name = 'dense_h1')(Y_)   
    Y_ = Dense(filters//2, activation = 'tanh', name = 'dense_h2')(Y_)
    Y_ = Dense(filters, activation = 'linear', name = 'dense_out')(Y_)
    Y_ = SAAF(break_points=25, break_range=0.2, magnitude=100, order=2, tied_feamap=True,
            kernel_initializer = 'random_normal', name = 'saaf_out')(Y_)
    
    Y_ = se_block(Y_, filters, weight_decay=0., amplifying_ratio=16, idx = 1)
    Y = Add(name='addition')([Y,Y_])
    Y = deconv(Y)
    
    model = Model(inputs=[x], outputs=[Y])
    
    model.compile(loss={'deconv': 'mae'},
                        loss_weights={'deconv': 1.0},
                        optimizer=Adam(lr=learning_rate))
    


    return model



# models that use Wavenet (WaveNet and CWAFx)


'''WaveNet code based on
[1] A Wavenet For Speech Denoising, 
Rethage, D.; Pons, J.; Serra, X.,
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.

Available in https://github.com/drethage/speech-denoising-wavenet
 '''


def dilated_residual_block(data_x, res_block_i, layer_i, dilation, stack_i, config,
                           num_residual_blocks, input_length, samples_of_interest_indices,
                           padded_target_field_length, context=True):
    
    

    original_x = data_x
    bias = True

    # Data sub-block
    data_out = keras.layers.Conv1D(2 * config['model']['filters']['depths']['res'],
                                                config['model']['filters']['lengths']['res'],
                                                dilation_rate=dilation, padding='same',
                                                use_bias=bias,
                                                name='res_%d_dilated_conv_d%d_s%d' % (
                                                res_block_i, dilation, stack_i),
                                                activation=None)(data_x)
    
    data_out_1 = Slice(
            (Ellipsis, slice(0, config['model']['filters']['depths']['res'])),
            (input_length, config['model']['filters']['depths']['res']),
            name='res_%d_data_slice_1_d%d_s%d' % (num_residual_blocks, dilation, stack_i))(data_out)
    
    data_out_2 = Slice(
            (Ellipsis, slice(config['model']['filters']['depths']['res'],
                             2 * config['model']['filters']['depths']['res'])),
            (input_length, config['model']['filters']['depths']['res']),
            name='res_%d_data_slice_2_d%d_s%d' % (num_residual_blocks, dilation, stack_i))(data_out)
   
   
    
    tanh_out = keras.layers.Activation('tanh')(data_out_1)
    sigm_out = keras.layers.Activation('sigmoid')(data_out_2)
    
    data_x = keras.layers.Multiply(name='res_%d_gated_activation_%d_s%d'
                                   % (res_block_i, layer_i, stack_i))([tanh_out, sigm_out])

    data_x = keras.layers.Conv1D(config['model']['filters']['depths']['res']
                                 + config['model']['filters']['depths']['skip'],
                                1,
                                padding='same',
                                use_bias = bias,
                                name='res_%d_conv_d%d_s%d' % (
                                res_block_i, dilation, stack_i))(data_x)
    
    res_x = Slice((Ellipsis, slice(0, config['model']['filters']['depths']['res'])),
                             (input_length, config['model']['filters']['depths']['res']),
                         name='res_%d_data_slice_3_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)
    
    skip_x = Slice((Ellipsis, slice(config['model']['filters']['depths']['res'],
                                               config['model']['filters']['depths']['res'] +
                                               config['model']['filters']['depths']['skip'])),
                              (input_length, config['model']['filters']['depths']['skip']),
                          name='res_%d_data_slice_4_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)
    
    if context == False:
        samples_of_interest_indices[0] = 0
        samples_of_interest_indices[-1] = config['model']['input_length']#k['win_length']
    
    skip_x = Slice((slice(samples_of_interest_indices[0], samples_of_interest_indices[-1], 1),
                               Ellipsis),
                   (padded_target_field_length, config['model']['filters']['depths']['skip']),
                   name='res_%d_keep_samples_of_interest_d%d_s%d' % (res_block_i, dilation, stack_i))(skip_x)
    
    res_x = keras.layers.Add()([original_x, res_x])
    

    return res_x, skip_x 


def wavenet(data_input, config, contextFrames=0, output_channels=1, context=True):

    

    num_residual_blocks = len(config['model']['dilations']) * config['model']['num_stacks']
    
    input_length = config['model']['input_length']
    if input_length == None:
        input_length = config['model']['target_field_length']*(contextFrames + 1)
    
    target_field_length = config['model']['target_field_length']
    half_target_field_length = target_field_length // 2
    target_padding = config['model']['target_padding']
    target_sample_index = int(np.floor(input_length / 2.0))
    samples_of_interest_indices = range(target_sample_index - half_target_field_length - target_padding,
                                        target_sample_index + half_target_field_length + target_padding + 1)
    padded_target_field_length = target_field_length + 2 * target_padding
   
    
    data_out = keras.layers.Conv1D(config['model']['filters']['depths']['res'],
                                                  config['model']['filters']['lengths']['res'], padding='same',
                                                  use_bias=True, name='initial_causal_conv')(data_input)
    
    skip_connections = []
    res_block_i = 0
    for stack_i in range(config['model']['num_stacks']):
        layer_in_stack = 0
        for dilation in config['model']['dilations']:
            res_block_i += 1
            data_out, skip_out = dilated_residual_block(data_out,
                                                        res_block_i,
                                                        layer_in_stack,
                                                        dilation,
                                                        stack_i,
                                                        config, 
                                                        num_residual_blocks, 
                                                        input_length, 
                                                        samples_of_interest_indices, 
                                                        padded_target_field_length,
                                                       context=context)
            if skip_out is not None:
                skip_connections.append(skip_out)
            layer_in_stack += 1

    skip_connections = keras.layers.Lambda(lambda inputs: tf.convert_to_tensor(inputs))(skip_connections)        

    skip_connections = keras.layers.Lambda(lambda inputs: tf.keras.backend.sum(inputs,
                                                                               axis=0,
                                                                               keepdims=False))(skip_connections)   

    data_out = keras.layers.Activation('relu')(skip_connections)

    data_out = keras.layers.Conv1D(config['model']['filters']['depths']['final'][0],
                                          config['model']['filters']['lengths']['final'][0], padding='same',
                                                  use_bias=True, name='penultimate_conv_1d')(data_out)

    
    data_out = keras.layers.Activation('relu')(data_out)

    data_out = keras.layers.Conv1D(config['model']['filters']['depths']['final'][1],
                                          config['model']['filters']['lengths']['final'][1], padding='same',
                                                  use_bias=True, name='final_conv_1d')(data_out)


    data_out = keras.layers.Conv1D(output_channels, 1)(data_out)
    
    return data_out



def WaveNet(learning_rate, wavenetConfig):
    
    data_input = keras.engine.Input(shape=(wavenetConfig['model']['input_length'], 1), name='data_input')

    
    data_out = wavenet(data_input, wavenetConfig)
    
    model = Model(inputs=[data_input], outputs=[data_out])     
        
    model.compile(loss='mae',
                  optimizer=Adam(lr=learning_rate))
    
    return model



def CWAFx(win_length, filters, kernel_size_1, learning_rate, wavenetConfig):
    
    kContext = 4 # past and subsequent frames
    
    x = Input(shape=(kContext*2+1, win_length, 1), name='input')
    
    conv = Conv1D(filters, kernel_size_1, strides=1, padding='same',
                       kernel_initializer='lecun_uniform', input_shape=(win_length, 1))
    
    activation_abs = Activation(K.abs)
    activation_sp = Activation('softplus')
    max_pooling = MaxPooling1D(pool_size=win_length//64)

    conv_smoothing = Conv1D_local(filters, kernel_size_1*2, strides=1, padding='same',
                                  kernel_initializer='lecun_uniform')
    
    
    deconv = Conv1D_tied(1, kernel_size_1, conv, padding='same', name='deconv')
     
        
    X = TimeDistributed(conv, name='conv')(x)
    X_abs = TimeDistributed(activation_abs, name='conv_activation')(X)
    M = TimeDistributed(conv_smoothing, name='conv_smoothing')(X_abs)
    M = TimeDistributed(activation_sp, name='conv_smoothing_activation')(M)
    P = X
    Z = TimeDistributed(max_pooling, name='max_pooling')(M)
    Z = Lambda(lambda inputs: tf.unstack(inputs, num=kContext*2+1, axis=1, name='unstack2'))(Z)
    Z = Concatenate(name='concatenate', axis=-2)(Z)
    
    Z = wavenet(Z, wavenetConfig, contextFrames=kContext, output_channels=filters, context=True)
  
    Z = Lambda((toPermuteDimensions), name='perm_1')(Z)
    Z = Dense(win_length//64, activation = 'tanh', name = 'dense_wn')(Z)
    Z = Lambda((toPermuteDimensions), name='perm_2')(Z)
    
    M_ = UpSampling1D(size=win_length//64, name='up_sampling_naive')(Z)
    P = Lambda(lambda inputs: tf.unstack(inputs, num=kContext*2+1, axis=1, name='unstack'))(P)
    Y = Multiply(name='phase_unpool_multiplication')([P[kContext],M_])

    Y_ = Dense(filters, activation = 'tanh', name = 'dense_in')(Y)
    Y_ = Dense(filters//2, activation = 'tanh', name = 'dense_h1')(Y_)   
    Y_ = Dense(filters//2, activation = 'tanh', name = 'dense_h2')(Y_)
    Y_ = Dense(filters, activation = 'linear', name = 'dense_out')(Y_)
 
    Y_ = SAAF(break_points=25, break_range=0.2, magnitude=100, order=2, tied_feamap=True,
            kernel_initializer = 'random_normal', name = 'saaf_out')(Y_)
    
    Y_ = se_block(Y_, filters, weight_decay=0., amplifying_ratio=16, idx = 1)
    
    Y = Add(name='addition')([Y,Y_])
    
    Y = deconv(Y)
    
    model = Model(inputs=[x], outputs=[Y])
    
    model.compile(loss={'deconv': 'mae'},
                        loss_weights={'deconv': 1.0},
                        optimizer=Adam(lr=learning_rate))
    

    return model





