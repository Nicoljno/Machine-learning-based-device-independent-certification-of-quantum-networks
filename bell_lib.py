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
    filename = 'matrici_mes_constraints_bell'
    with open(filename, 'rb') as f:
        mes_diz = pickle.load(f)
    filename = 'matrici_non_mes_constraints_bell'
    with open(filename, 'rb') as g:
        non_mes_diz = pickle.load(g)
    parties = ['A','B']
    inputs = ['0','1']
    loss_=[]
    keys=list(non_mes_diz.keys())
    for _ in range(1):
        CHSH=(expected_value(y_pred[0][0:16], 'A0B0', parties, inputs)+
              expected_value(y_pred[0][0:16], 'A0B1', parties, inputs)+
              expected_value(y_pred[0][0:16], 'A1B0', parties, inputs)-
              expected_value(y_pred[0][0:16], 'A1B1', parties, inputs))
        gamma=(sum(mes_diz[i]*expected_value(y_pred[0][0:16], i, parties, inputs) for i in mes_diz.keys() )+
               sum(non_mes_diz[keys[i]]*(y_pred[_][16+i]+train[_][i]) for i in range(len(keys)) )
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
    return tf.cond(tf.math.greater(no_sign, 0.001), lambda: no_sign, lambda: tf.cond(tf.math.greater(K.sum(loss_), 0), lambda: K.sum(loss_), lambda: K.sum(loss_)+K.sum(CHSH)))


visible = Input(shape=(30,))
hidden1 = Dense(300, activation = 'elu')(visible)
hidden2 = Dense(300, activation = 'elu')(hidden1)
hidden3 = Dense(300, activation = 'elu')(hidden2)
hidden4 = Dense(300, activation = 'elu')(hidden3)
hidden5 = Dense(300, activation = 'elu')(hidden4)
hidden6 = Dense(300, activation = 'elu')(hidden5)
hidden7 = Dense(300, activation = 'elu')(hidden6)
output1 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output2 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output3 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output4 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output5 = Dense(30, activation = 'tanh')(hidden7)
output=tf.concat([output1, output2,output3,output4, output5], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.00001, momentum=0.8)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)



input_=np.zeros((1000000,30))
train=tf.constant(input_)

model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0)


model.save('bell_lib')
