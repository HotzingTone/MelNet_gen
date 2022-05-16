import tensorflow as tf
from tensorflow.python.framework.config import list_physical_devices
from mel_data_source import MelDataSource
from model import Model
from trainer import Trainer
from data import create_dataset


# print(f'\nTensorFlow ver. {tf.__version__}')
# if tf.__version__[0] == '1':
#   tf.enable_v2_behavior()
#   print('Warning: you should use TensorFlow 2')
# print('\n', list_physical_devices('GPU'), '\n')
#
# data_source = MelDataSource('./CodeLab_TestData')
# n_epochs = 10
#
# # Baseline uses a multivariate Gaussian that models all log-mel parameters in one frame jointly
# model_baseline = Model(mode='baseline', n_bins=data_source.mel_dim)
# Trainer('train_reports/baseline', data_source, model_baseline).run(n_epochs)
#
# # MelNet uses Gaussian Mixture with K components that models log-mel parameters for each bin
# model_melnet = Model(k_mix=4, n_bins=data_source.mel_dim)
# Trainer('train_reports/melnet', data_source, model_melnet).run(n_epochs)

dataset = create_dataset('./CodeLab_TestData')
for i, d in enumerate(dataset):
    print(i, d.shape)
