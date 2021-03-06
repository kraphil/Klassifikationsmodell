!rm -rf ./logs/ 
!pip install silence-tensorflow

import silence_tensorflow.auto
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import ParameterGrid
import tensorflow as tf

import importlib
import argparse
import os, sys
import argparse
import pandas as pd
import numpy as np
import pickle
import time


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve

# %matplotlib inline

from torch.autograd import Variable

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector, Bidirectional, Reshape, TimeDistributed, concatenate, Flatten, Activation, Dot, BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import objectives
from keras.utils.vis_utils import plot_model

sys.path.insert(0, 'utils.py')
sys.path.insert(0, 'models.py')
from utils import *
from models import *

#name = 'bpi_2012'
#name = 'small_log'
#name = 'large_log'
name = 'bpic_2013'

"""
parser = {
    'train': True,
    'test': True,
    'model_class': 'AE',
    'model_name': '',
    'data_dir': '../data/',
    'data_file': name + '.csv',
    'anomaly_pct': 0.1,
    #'input_dir': '../input/{}/'.format(name), 
    'output_dir': './output/{}/'.format(name),
    'scaler': 'standardization',
    'batch_size' : 16,
    'epochs' : 10,
    'no_cuda' : False,
    'seed' : 7,
    'layer1': 1000,
    'layer2': 100,
    'lr': 0.002,
    'betas': (0.9, 0.999),   
    'lr_decay': 0.90,
}
"""

#args = argparse.Namespace(**parser)

preprocessed_data_name = os.path.join('preprocessed_data_{}.pkl'.format(args.anomaly_pct))
with open(preprocessed_data_name, 'rb') as f:
    input_train = pickle.load(f)
    input_val = pickle.load(f)
    input_test = pickle.load(f)
    pad_index_train = pickle.load(f)
    pad_index_val = pickle.load(f)
    pad_index_test = pickle.load(f)
    activity_label_test = pickle.load(f)
    activity_label_val = pickle.load(f)
    time_label_val = pickle.load(f)
    time_label_test = pickle.load(f)
    train_case_num = pickle.load(f)
    val_case_num = pickle.load(f)
    test_case_num = pickle.load(f)
    train_row_num = pickle.load(f)
    val_row_num = pickle.load(f)
    test_row_num = pickle.load(f)
    min_value = pickle.load(f)
    max_value = pickle.load(f)
    mean_value = pickle.load(f)
    std_value = pickle.load(f)
    cols = pickle.load(f)
    statistics_storage = pickle.load(f)
    true_time = pickle.load(f)
    true_act = pickle.load(f)
    full_true_time = pickle.load(f)
    full_true_act = pickle.load(f)

data_df = pd.DataFrame({'ActivityLabel': activity_label_test,
                              'TimeLabel': time_label_test})
data_df.head()

input_train.shape

"""Modell"""

def classificationModel(params, log_dir, id):
  #Parameter
  batch_size = params['batch_size']
  #epochs = params['epochs']
  activation = params['activation']
  learning_rate = params['learning_rate']
  #dropout_rate = params['dropout_rate']
  
  #Eingabe
  #Variablen
  timesteps = input_train.shape[1]
  input_dim = input_train.shape[2]
  latent_dim = 100 
  z_dim =2
  epsilon_std=1

  #Input
  inputs = Input(shape=(timesteps, input_dim))

  #Encoder Bidirectional LSTM
  encoder_stack_h, encoder_last_h, encoder_last_c, *_ = Bidirectional(LSTM(latent_dim, activation="relu", return_state=True, return_sequences=True))(inputs)
  encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
  encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)
  print(encoder_stack_h)
  print(encoder_last_h)

  #Variational Layer
  z_mean = Dense(z_dim)(encoder_stack_h)
  print(z_mean)
  z_log_sigma = Dense(z_dim)(encoder_stack_h)
  print(z_log_sigma)

  def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0],z_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

  z = Lambda(sampling)([z_mean, z_log_sigma])

  #Decoder Bidirectional LSTM
  decoder = RepeatVector(timesteps)(encoder_last_h)
  decoder = Bidirectional(LSTM(latent_dim, activation="relu", return_sequences=True))(decoder)
  print(decoder)

  #Self-Attention Layer
  attention = keras.layers.dot([decoder, z], axes=[1,1])
  attention = Activation('softmax')(attention)
  print(attention))

  context = keras.layers.dot([z, attention], axes=[2,2])
  print(context)
  decoder_combined_context = concatenate([context, decoder])

  #Output
  output = TimeDistributed(Dense(input_dim, activation='sigmoid'))(decoder_combined_context)
  lstmae = Model(inputs, output)

  #Loss Function
  def vae_loss(inputs, decoder_combined_context):
        xent_loss = objectives.mse(inputs, decoder_combined_context)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

  #Modell bauen
  lstmae.compile(optimizer=Adam(learning_rate=0.006), loss=vae_loss)
  lstmae.summary()
  plot_model(lstmae, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  checkpointer = ModelCheckpoint(filepath="model_seqs2.h5",
                              verbose=0,
                              save_best_only=True)

  tensorboard = TensorBoard(log_dir='logs/' + run_name,
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

  history = lstmae.fit(input_train,input_train, 
                       epochs=20,
                       batch_size=batch_size, 
                       verbose=1, 
                       validation_data=(input_test, input_test), 
                       callbacks=[hp.KerasCallback(log_dir, params, trial_id=id), checkpointer, tensorboard]).history
  keras.backend.clear_session()

param_dict = {'batch_size': np.arange(start=64, stop=256, step=64),
              'activation': ['relu', 'sigmoid', 'softmax'],
              'learning_rate': np.arange(start=0.001, stop=0.007, step=0.005)}


grid = ParameterGrid(param_dict)

session = 0

for params in grid:
  run_name = "run-%d" % session
  print('--- Trial: %s' % run_name)
  print({p: params[p] for p in params})


  
  classificationModel(params = params, 
        log_dir = 'logs/' + run_name,
        id = run_name)
  
  session += 1

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# %tensorboard --logdir logs/
