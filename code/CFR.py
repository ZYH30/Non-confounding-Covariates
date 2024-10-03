# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Lambda, Input, Dense,BatchNormalization,Concatenate,Add,Subtract
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy,CosineSimilarity,cosine_similarity,sparse_categorical_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os
import pandas as pd
from graphviz import Digraph
import random

import warnings
warnings.filterwarnings('ignore') 

import os 

import tensorflow as tf

def mmd2_lin(X,t,p):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc,0)
    mean_treated = tf.reduce_mean(Xt,0)

    mmd = tf.reduce_sum(tf.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def CFR(intermediate_x_dim = 20,intermediate_dim = 10,t_dim = 1,c_dim = 25,y_dim = 1):
    
    inputs_t = Input(shape=(t_dim,), name='input_t_balance')
    inputs_c = Input(shape=(c_dim,), name='input_c_balance')
    inputs_y = Input(shape=(y_dim,), name='input_y_balance')
    

    x_rep_ = Dense(intermediate_x_dim, activation='leaky_relu')(inputs_c)
    x_rep = Dense(round(intermediate_x_dim/2), activation='leaky_relu')(x_rep_)
    
    # split head
    i0 = tf.cast(tf.where(inputs_t < 1)[:,0], tf.int32)
    i1 = tf.cast(tf.where(inputs_t > 0)[:,0], tf.int32)

    rep0 = tf.gather(x_rep, i0)
    rep1 = tf.gather(x_rep, i1)

    x_predtic_y0_ = Dense(intermediate_dim, activation='leaky_relu')(rep0)
    x_predtic_y0 = Dense(y_dim,name = 'predtic_t_of_y0')(x_predtic_y0_)

    x_predtic_y1_ = Dense(intermediate_dim, activation='leaky_relu')(rep1)
    x_predtic_y1 = Dense(y_dim,name = 'predtic_t_of_y1')(x_predtic_y1_)

    x_predtic_y = tf.dynamic_stitch([i0, i1], [x_predtic_y0, x_predtic_y1])
    
    cfr = Model([inputs_t, inputs_c,inputs_y], [x_rep, x_predtic_y], name='cfr')
    
    prediction_loss_y = mse(inputs_y, x_predtic_y)
    balance_loss = mmd2_lin(x_rep,inputs_t,p = 0.5)
    cfr_loss = K.mean(prediction_loss_y + balance_loss)

    cfr.add_loss(cfr_loss)
    cfr.compile(optimizer = 'adam')
    
    return cfr

def splitData(data,X = ['C'],T = 'T',Y = 'Y',seed = 1):
    
    train_val_rate = 0.8
    train_val_samples = round(len(data) * train_val_rate)
    
    random.seed(seed)
    train_val_sample_select = random.sample(range(len(data)), train_val_samples)
    test_sample_select = list(set(range(len(data))) - set(train_val_sample_select))

    input_t = data.loc[train_val_sample_select,T].reset_index(drop=True)
    input_c = data.loc[train_val_sample_select,X].reset_index(drop=True)
    input_y = data.loc[train_val_sample_select,Y].reset_index(drop=True)
    mu0 = data.loc[train_val_sample_select,'Y_0'].reset_index(drop=True)
    mu1 = data.loc[train_val_sample_select,'Y_1'].reset_index(drop=True)

    input_t_test = data.loc[test_sample_select,T].reset_index(drop=True)
    input_c_test = data.loc[test_sample_select,X].reset_index(drop=True)
    input_y_test = data.loc[test_sample_select,Y].reset_index(drop=True)
    mu0_test = data.loc[test_sample_select,'Y_0'].reset_index(drop=True)
    mu1_test = data.loc[test_sample_select,'Y_1'].reset_index(drop=True)

    return input_t,input_c,input_y,mu0,mu1,input_t_test,input_c_test,input_y_test,mu0_test,mu1_test

def Eveluation(input_t,input_y,factual_y,predict_y,factual_predict,is_mu = False):
    
    t_coff = [i if i == 1 else -1 for i in input_t]
    if is_mu:
        ite = factual_y - input_y
    else:
        ite = [(input_y[i] - factual_y[i]) * t_coff[i] for i in range(len(input_t))]
    predict_ite = [(predict_y[i] - factual_predict[i]) * t_coff[i] for i in range(len(input_t))]
    
    pehe = np.sqrt(np.mean(np.power(np.array(predict_ite).reshape(len(predict_ite),) - ite,2)))
    abs_ate = np.abs(np.mean(predict_ite) - np.mean(ite))
    
    return pehe, abs_ate
    
def pipiline(data, epochs = 300, batch_size = 256, c_dim = 1,seed = 23):

    input_t,input_c,input_y,mu0,mu1,input_t_test,input_c_test,input_y_test,mu0_test,mu1_test = data
    
    cfr = CFR(intermediate_x_dim = c_dim * 4,intermediate_dim = c_dim,t_dim = 1,c_dim = c_dim,y_dim = 1)
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cfr_history = cfr.fit([input_t,input_c,input_y],epochs=epochs,batch_size=batch_size,validation_split = 0.2,verbose = 0)
    
    cfr_predict = cfr.predict([input_t,input_c,input_y])
    predict_y = cfr_predict[1]
    
    factual_t = 1 - input_t
    factual_predict = cfr.predict([factual_t,input_c,input_y])
    factual_predict = factual_predict[1]
    
    pehe,abs_ate = Eveluation(input_t,mu0,mu1,predict_y,factual_predict,is_mu = True)
    
    cfr_predict_test = cfr.predict([input_t_test,input_c_test,input_y_test])
    predict_y_test = cfr_predict_test[1]
    
    factual_t_test = 1 - input_t_test
    factual_predict_test = cfr.predict([factual_t_test,input_c_test,input_y_test])
    factual_predict_test = factual_predict_test[1]
    
    pehe_test,abs_ate_test = Eveluation(input_t_test,mu0_test,mu1_test,predict_y_test,factual_predict_test,is_mu = True)
    
    return pehe,abs_ate,pehe_test,abs_ate_test

dataFrame = pd.read_csv('../data/data-discrete.csv')
# C
print("================C==================")
X = ['C']
c_dim = len(X)
dataS = splitData(dataFrame, X = X)
seed = 20
peheG,abs_ateG,peheG_test,abs_ateG_test = pipiline(data = dataS, epochs = 500, batch_size = 256, c_dim = c_dim, seed = seed)  
print("peheG:{},abs_ateG:{},peheG_test:{},abs_ateG_test:{}".format(peheG,abs_ateG,peheG_test,abs_ateG_test))

# C+M
print("================C+M==================")
X = ['C','M']
c_dim = len(X)
dataS = splitData(dataFrame, X = X)
seed = 15
peheG,abs_ateG,peheG_test,abs_ateG_test = pipiline(data = dataS, epochs = 400, batch_size = 256, c_dim = c_dim, seed = seed)  
print("peheG:{},abs_ateG:{},peheG_test:{},abs_ateG_test:{}".format(peheG,abs_ateG,peheG_test,abs_ateG_test))

# C+M+Z
print("================C+M+Z==================")
X = ['C','M','Z']
c_dim = len(X)
dataS = splitData(dataFrame, X = X)
seed = 10
peheG,abs_ateG,peheG_test,abs_ateG_test = pipiline(data = dataS, epochs = 400, batch_size = 256, c_dim = c_dim, seed = seed)  
print("peheG:{},abs_ateG:{},peheG_test:{},abs_ateG_test:{}".format(peheG,abs_ateG,peheG_test,abs_ateG_test))

# C+M+Z+I
print("================C+M+Z+I==================")
X = ['C','M','I','Z']
c_dim = len(X)
dataS = splitData(dataFrame, X = X)
seed = 10
peheG,abs_ateG,peheG_test,abs_ateG_test = pipiline(data = dataS, epochs = 400, batch_size = 256, c_dim = c_dim, seed = seed)  
print("peheG:{},abs_ateG:{},peheG_test:{},abs_ateG_test:{}".format(peheG,abs_ateG,peheG_test,abs_ateG_test))

print("================C+X==================")
X = ['C','X']
c_dim = len(X)
dataS = splitData(dataFrame, X = X)
seed = 46
peheG,abs_ateG,peheG_test,abs_ateG_test = pipiline(data = dataS, epochs = 500, batch_size = 256, c_dim = c_dim, seed = seed)  
print("peheG:{},abs_ateG:{},peheG_test:{},abs_ateG_test:{}".format(peheG,abs_ateG,peheG_test,abs_ateG_test))


# C+X+M
print("================C+X+M==================")
X = ['C','M','X']
c_dim = len(X)
dataS = splitData(dataFrame, X = X)
seed = 10
peheG,abs_ateG,peheG_test,abs_ateG_test = pipiline(data = dataS, epochs = 400, batch_size = 256, c_dim = c_dim, seed = seed)  
print("peheG:{},abs_ateG:{},peheG_test:{},abs_ateG_test:{}".format(peheG,abs_ateG,peheG_test,abs_ateG_test))


# C+X+M+Z
print("================C+X+M+Z==================")
X = ['C','M','X','Z']
c_dim = len(X)
dataS = splitData(dataFrame, X = X)
seed = 10
peheG,abs_ateG,peheG_test,abs_ateG_test = pipiline(data = dataS, epochs = 400, batch_size = 256, c_dim = c_dim, seed = seed)  
print("peheG:{},abs_ateG:{},peheG_test:{},abs_ateG_test:{}".format(peheG,abs_ateG,peheG_test,abs_ateG_test))


# C+X+M+Z+I
print("================C+X+M+Z+I==================")
X = ['C','M','X','Z','I']
c_dim = len(X)
dataS = splitData(dataFrame, X = X)
seed = 10
peheG,abs_ateG,peheG_test,abs_ateG_test = pipiline(data = dataS, epochs = 400, batch_size = 256, c_dim = c_dim, seed = seed)  
print("peheG:{},abs_ateG:{},peheG_test:{},abs_ateG_test:{}".format(peheG,abs_ateG,peheG_test,abs_ateG_test))


saveResultG = pd.DataFrame(columns = ['exp_ind','split_seed','X','peheG','abs_ateG','peheG_test','abs_ateG_test'])
for i in range(100):
    print("outer run {} times!".format(i))
    X_set = [['C'],['C','M'],['C','X'],['C','M','X'],['C','M','Z'],['C','X','Z']]
    for j in X_set:
        X = j
        c_dim = len(X)
        for k in [220422,230423,240424]:
            dataS = splitData(dataFrame, X = X,seed = k)
            seed = i
            peheG,abs_ateG,peheG_test,abs_ateG_test = pipiline(data = dataS, epochs = 400, batch_size = 256, c_dim = c_dim, seed = seed)  
            data_G_ = pd.DataFrame({'exp_ind':i, 'split_seed':k, 'X':'+'.join(j),
                                    'peheG':peheG,'abs_ateG':abs_ateG,
                                    'peheG_test':peheG_test,'abs_ateG_test':abs_ateG_test},index = [0])
            saveResultG = saveResultG.append(data_G_,ignore_index=True)
            

import time

T = round(time.time())  
saveResultG.to_csv('../data/IHDPsaveResultG_{}.csv'.format(T),index = False)    

saveResultG.describe()
