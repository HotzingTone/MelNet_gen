import tensorflow as tf
from tensorflow.python.framework.config import list_physical_devices
from mel_data_source import MelDataSource
from model import Model
from trainer import Trainer
from data import DataSource


# print(f'\nTensorFlow ver. {tf.__version__}')
# if tf.__version__[0] == '1':
#   tf.enable_v2_behavior()
#   print('Warning: you should use TensorFlow 2')
# print('\n', list_physical_devices('GPU'), '\n')


def train_once():
    source = DataSource('./CodeLab_TestData')
    model = Model(k_mix=4)
    n_epochs = 1
    Trainer('train_reports/melnet', source, model).run(n_epochs)

def gen(length):
    model = Model(k_mix=4)
    return model.generate(length)

def check_data():
    source = DataSource('./CodeLab_TestData')
    data = source.get_data()
    for i, X in enumerate(data):
        print()
        print(i)
        for tier in X:
            print(tier['even'].shape, tier['odd'].shape)

# train_once()

sample = gen(3)
print(sample.shape)

# check_data()