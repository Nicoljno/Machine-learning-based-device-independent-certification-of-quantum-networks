import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from math import *
from tensorflow import keras
from tensorflow import concat
from tensorflow.keras.callbacks import LambdaCallback, Callback
from tensorflow.math import multiply, reduce_sum, reduce_min
from tensorflow.keras import layers, initializers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import sys


def expected_value(p, operator, parties, inputs):
    n=2
    suffix_operators={}
    suffix=0
    e_value=0
    inputs_voc={}
    for i in range(len(inputs)):
        inputs_voc[inputs[i]]=i 
    for i in range(len(parties)):
        suffix_operators[parties[len(parties)-1-i]]=i
    list_operators = [operator[j:j+1] for j in range(0, len(operator), n) ]
    list_inputs = [operator[j+1:j+2] for j in range(0, len(operator), n) ]
    if(len(inputs)>1):
        for i in range(len(list_operators)):
            suffix+=2**(len(parties)+suffix_operators[list_operators[i]])*inputs_voc[list_inputs[i]]
    for i in range(2**len(parties)):
        binar=bin(i+2**(len(parties)))[3:]
        coeff = 0
        for j in list_operators:
            coeff+=int(binar[len(parties)-1-suffix_operators[j]])
        e_value+=(-1)**(coeff)*p[suffix+i]
    return e_value


def custom_loss(train, y_pred):
    filename = 'matrici_mes_constraints_instrumental_lev_1'
    with open(filename, 'rb') as f:
        mes_diz = pickle.load(f)
    filename = 'matrici_non_mes_constraints_instrumental_lev_1'
    with open(filename, 'rb') as g:
        non_mes_diz = pickle.load(g)
    parties = ['A','B']
    inputs = ['0','1','2']
    loss_=[]
    keys=list(non_mes_diz.keys())
    '''
    #errori asuma dati generati numericamente N=10^6
    errors=np.array([0.018709988592, 0.018709988592, 0.018709988592, 0.018709988592,
                     0.018709988592, 0.018709988592, 0.018709988592, 0.018709988592,
                     0.018709988592, 0.018709988592, 0.018709988592, 0.018709988592,
                     0.018709988592, 0.018709988592, 0.018709988592, 0.018709988592,
                     0.026459919618, 0.026459919618, 0.026459919618, 0.026459919618,
                     0.026459919618, 0.026459919618, 0.026459919618, 0.026459919618])
    errors=errors*sqrt(-log(10**(-5)*0.5))/sqrt(-log(10**(-1)*0.5))
    '''
    
    #errori hoeffding dati generati numericamente
    errors=np.array([0.038090232, 0.038090232, 0.038090232, 0.038090232,
                     0.038090232, 0.038090232, 0.038090232, 0.038090232,
                     0.038090232, 0.038090232, 0.038090232, 0.038090232,
                     0.038090232, 0.038090232, 0.038090232, 0.038090232,
                     0.053867722, 0.053867722, 0.053867722, 0.053867722,
                     0.053867722, 0.053867722, 0.053867722, 0.053867722])
    errors=errors*sqrt(5000)/sqrt(30000)
    '''
    #errori hoeffding dati sperimentali
    errors=np.array([0.0110501283, 0.0110501283, 0.0110501283, 0.0110501283,
                     0.00618567728, 0.00618567728, 0.00618567728, 0.00618567728,
                     0.00745087848, 0.00745087848, 0.00745087848, 0.00745087848,
                     0.00987155205, 0.00987155205, 0.00987155205, 0.00987155205,
                     0.0148173167, 0.0148173167, 0.0148173167, 0.0148173167,
                     0.0148173167, 0.0148173167, 0.0148173167, 0.0148173167])
    #errors=errors*sqrt(-log(10**(-40)*0.5))/sqrt(-log(10**(-1)*0.5))
    '''

    p=train[0][0:24]
    p=K.abs(p+tf.multiply(y_pred[0][-24:], errors.T))
    P1=p[0:4]/K.sum(p[0:4])
    P2=p[4:8]/K.sum(p[4:8])
    P3=p[8:12]/K.sum(p[8:12])
    P4=p[12:16]/K.sum(p[12:16])
    P5=p[16:20]/K.sum(p[16:20])
    P6=p[20:24]/K.sum(p[20:24])
    p=tf.concat([P1,P2,P3,P4,P5,P6], axis=0)

    for _ in range(1):
        CHSH=(expected_value(p, 'A0B0', parties, inputs)+
              expected_value(p, 'A0B1', parties, inputs)+
              expected_value(p, 'A1B0', parties, inputs)-
              expected_value(p, 'A1B1', parties, inputs))
        gamma=(sum(mes_diz[i]*expected_value(p, i, parties, inputs) for i in mes_diz.keys() )+
               sum(non_mes_diz[keys[i]]*(y_pred[_][-(24+i)]) for i in range(len(keys)) )
               )
        loss_.append(-reduce_min(tf.linalg.eigvalsh(gamma)))
    loss_=tf.stack(loss_)
    no_sign=0
    vals=[0,1]

    for b in vals:
        for y in vals:
            no_sign+=abs(sum(y_pred[0][a*2+b+y*4]-y_pred[0][a*2+b+8+y*4] for a in vals))
    for a in vals:
        for x in vals:
            no_sign+=abs(sum(y_pred[0][a*2+b+x*8]-y_pred[0][a*2+b+x*8+4] for b in vals))
    tf.print(CHSH, K.sum(loss_), no_sign)
    return tf.cond(tf.math.greater(no_sign, 0.01), lambda: no_sign, lambda: tf.cond(tf.math.greater(K.sum(loss_), 0), lambda: K.sum(loss_), lambda: K.sum(loss_)-K.sum(CHSH)))


print_weights = LambdaCallback(on_batch_end=lambda batch, logs: print(logs["loss"]))

visible = Input(shape=(24,))
hidden1 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(visible)
hidden2 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden1)
hidden3 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden2)
hidden4 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden3)
hidden5 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden4)
hidden6 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden5)
hidden7 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden6)
output1 = Dense(4, activation = 'relu', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output2 = Dense(24, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output = tf.concat([output1, output2], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.0001, momentum=0.8)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)


input_=pd.read_table('bonet_samples.txt', delimiter=" ", header=None)
input_=input_.to_numpy()
input_=input_.astype('float32')

train=tf.constant(input_)

model.fit(train, train, batch_size=1, epochs=2, shuffle=True, verbose=0)

#model.fit(train, train, batch_size=20, epochs=4, shuffle=True, verbose=0, callbacks=[callback, LRchanger()])


model.save('hoeff_bonet_n30000')
