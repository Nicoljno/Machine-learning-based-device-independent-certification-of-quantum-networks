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
    part_diz = { 'A' : 3, 'B' : 2}
    parties = ['A','B']
    level = 2
    filename = f'matrici_mes_constraints_instrumental_lev_{level}'
    with open(filename, 'rb') as f:
        mes_diz = pickle.load(f)
    filename = f'matrici_non_mes_constraints_instrumental_lev_{level}'
    with open(filename, 'rb') as f:
        non_mes_diz = pickle.load(f)
    non_mes_keys={}
    mes_keys={}
    for i, j in enumerate(list(non_mes_diz.keys())):
        non_mes_keys[j]=i
    for i, j in enumerate(list(mes_diz.keys())):
        mes_keys[j]=i
    inputs = ['0','1','2']
    loss_=[]
    CHSH=[]
    vals=[0,1]
    non_mes_element=list(non_mes_diz.keys())
    mes_element=list(mes_diz.keys())
    p=y_pred[0][0:16]
    zeros=np.array([0,0])
    p=tf.concat([p,[y_pred[0][16]],[y_pred[0][17]],[y_pred[0][18]],[y_pred[0][19]],zeros,zeros], axis=0)
    #tf.print(p)
    for _ in range(1):
        CHSH.append(expected_value(p,'A0B0', parties, inputs)+
                    expected_value(p,'A1B0', parties, inputs)+
                    expected_value(p,'A0B1', parties, inputs)-
                    expected_value(p,'A1B1', parties, inputs))
#        gamma=(sum(mes_diz[mes_element[i]]*expected_value(p, mes_element[i], parties, inputs)
#                   for i in range(len(list(mes_keys.keys()))))+
#               sum(non_mes_diz[non_mes_element[i]]*(y_pred[_][20+i]+train[_][i]) for i in range(len(list(non_mes_keys.keys()))))
#               )
        gamma=(sum(mes_diz[i]*expected_value(p, i, parties, inputs) for i in mes_diz.keys() )+
               sum(non_mes_diz[non_mes_element[i]]*(y_pred[_][20+i]+train[_][i]) for i in range(len(non_mes_element)) )
               )
        loss_.append(-reduce_min(tf.linalg.eigvalsh(gamma)))
    tf.print(K.sum(CHSH), K.sum(loss_))
    no_sign=0
    vals=[0,1]
    for b in vals:
        for y in vals:
            no_sign+=abs(sum(y_pred[0][a*2+b+y*4]-y_pred[0][a*2+b+8+y*4] for a in vals))
    for a in vals:
        for x in vals:
            no_sign+=abs(sum(y_pred[0][a*2+b+x*8]-y_pred[0][a*2+b+x*8+4] for b in vals))
    return tf.cond(tf.math.greater(no_sign, 0.01), lambda: no_sign, lambda: tf.cond(tf.math.greater(K.sum(loss_), 0), lambda: K.sum(loss_), lambda: K.sum(loss_)+K.sum(CHSH)))
    #return tf.cond(tf.math.greater(K.sum(loss_), 0), lambda: K.sum(loss_), lambda: K.sum(loss_)+K.sum(CHSH))
    
class LRchanger(Callback):
    def __init__(self,display=20000):
        self.seen = 0
        self.display = display
    def on_batch_end(self,batch,logs={}):
        self.seen += 1
        #print(self.model.optimizer.lr)
        if self.seen % self.display == 0:
            self.model.optimizer.lr=self.model.optimizer.lr*0.1
            #print(self.model.optimizer.lr)

visible = Input(shape=(93,))
hidden1 = Dense(500, activation = 'elu')(visible)
hidden2 = Dense(500, activation = 'elu')(hidden1)
hidden3 = Dense(500, activation = 'elu')(hidden2)
hidden4 = Dense(500, activation = 'elu')(hidden3)
hidden5 = Dense(500, activation = 'elu')(hidden4)
hidden6 = Dense(500, activation = 'elu')(hidden5)
hidden7 = Dense(500, activation = 'elu')(hidden6)
output1 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output2 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output3 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output4 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output5 = Dense(2, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output6 = Dense(2, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output7 = Dense(93, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output=tf.concat([output1, output2, output3, output4, output5, output6, output7], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.0001, momentum=0.8)
#opt=keras.optimizers.Adamax(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)

#input_=np.zeros((20000,11))
input_=np.zeros((200000,93))
#input_=np.random.uniform(size=((20000,93)), low=-0.0001, high=0.0001)
train=tf.constant(input_)

model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0)#, callbacks=[LRchanger()])

level=2
model.save(f'instrumental_lib_lev_{level}')
