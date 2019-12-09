from sacred import Ingredient

config_ingredient = Ingredient("cfg")

@config_ingredient.config
def cfg():
    
    # models configuration
    
    pretraining = { 'epoch' : 1,
                    'filters' : 128,
                    'kernelSize' : 64,
                    'learningRate' : 0.0001,
                    'winLength' : 4096,
                    'modelsPath': './Models/',
                    'monitorLoss': 'val_loss'
                   }
    
    CAFx = { 'epoch' : 1,
                    'filters' : 128,
                    'kernelSize' : 64,
                    'learningRate' : 0.0001,
                    'winLength' : 4096,
                    'modelsPath': './Models/',
                    'monitorLoss': 'val_loss'
                   }
    
    CRAFx = { 'epoch' : 1,
                    'filters' : 32,
                    'kernelSize' : 64,
                    'learningRate' : 0.0001,
                    'winLength' : 4096,
                    'modelsPath': './Models/',
                    'monitorLoss': 'val_loss'
                   }
    
    CWAFx = { 'epoch' : 1,
                    'filters' : 32,
                    'kernelSize' : 64,
                    'learningRate' : 0.0001,
                    'winLength' : 4096,
                    'modelsPath': './Models/',
                    'monitorLoss': 'val_loss',
                    'wavenetConfig': {
                        'model': {
                        'dilations': [1, 2, 4, 8, 16, 32, 64],
                        'filters': {'depths': {'final': [32, 32],
                        'res': 32,
                        'skip': 32},
                        'lengths': {'final': [3, 3], 'res': 3, 'skip': 1}},
                        'input_length': 576,
                        'num_stacks': 2,
                        'target_field_length': 576,
                        'target_padding': 0}}
                   }
    
    WaveNet = { 'epoch' : 1,
                    'learningRate' : 0.0001,
                    'winLength' : 5118,
                    'modelsPath': './Models/',
                    'monitorLoss': 'val_loss',
                    'wavenetConfig': {
                        'model': {
                        'dilations': [1, 2, 4, 8, 16, 32, 64, 128],
                        'filters': {'depths': {'final': [2048, 256],
                          'res': 16,
                          'skip': 16},
                         'lengths': {'final': [3, 3], 'res': 3, 'skip': 1}},
                        'input_length': 5118,
                        'num_stacks': 2,
                        'target_field_length': 4096,
                        'target_padding': 0}}
                   }
    

    