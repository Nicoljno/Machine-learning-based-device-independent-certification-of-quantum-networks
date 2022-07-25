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

#bilocality
class LRchanger(Callback):
    def __init__(self,display=40000):
        self.seen = 0
        self.display = display
    def on_batch_end(self,batch,logs={}):
        self.seen += 1
        #print(self.model.optimizer.lr)
        if self.seen % self.display == 0:
            self.model.optimizer.lr=self.model.optimizer.lr*0.2
            #print(self.model.optimizer.lr)

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

def causal_indep_prod(y_pred, train, operator, parties, inputs, mes_diz, non_mes_diz, indep_cons_diz, non_mes_element):
    prod=list(operator.split('*'))
    tmp=1.
    for j in prod:
        if j in mes_diz.keys():
            tmp*=expected_value(y_pred[0:64], j, parties, inputs)
        elif j in non_mes_diz.keys():
            tmp*=y_pred[64+non_mes_element[j]]+train[non_mes_element[j]]
    return tmp


def custom_loss(train, y_pred):
    filename = 'matrici_mes_constraints_bilocality'
    with open(filename, 'rb') as f:
        mes_diz = pickle.load(f)
    filename = 'matrici_non_mes_constraints_bilocality'
    with open(filename, 'rb') as f:
        non_mes_diz = pickle.load(f)
    filename = 'matrici_indep_constraints_bilocality'
    with open(filename, 'rb') as f:
        indep_cons_diz=pickle.load(f)
    non_mes_element={}
    for i, j in enumerate(list(non_mes_diz.keys())):
        non_mes_element[j]=i
    parties = ['A','B','C']
    inputs = ['0','1']
    loss_=[]
    I=[]
    vals=[0,1]
    keys=list(non_mes_diz.keys())
    for _ in range(1):
        I1=0.25*sum((-1)**(a+b+c)*y_pred[_][x*32+8*y+a*4+b*2+c] for x in vals for y in vals for a in vals for b in vals for c in vals)
        I2=0.25*sum((-1)**(a+b+c+x+y)*y_pred[_][x*32+16+y*8+a*4+b*2+c] for x in vals for y in vals for a in vals for b in vals for c in vals)
        I.append(K.sqrt(K.abs(I1))+K.sqrt(K.abs(I2)))
        gamma=(sum(mes_diz[i]*expected_value(y_pred[0][0:64], i, parties, inputs) for i in mes_diz.keys() )+
               sum(non_mes_diz[keys[i]]*(y_pred[_][64+i]+train[_][i]) for i in range(len(keys)))+
               sum(causal_indep_prod(y_pred[0], train[0], i, parties, inputs, mes_diz, non_mes_diz, indep_cons_diz, non_mes_element)*indep_cons_diz[i]
                   for i in indep_cons_diz.keys())
               )
        loss_.append(-reduce_min(tf.linalg.eigvalsh(gamma)))
    tf.print(K.sum(I), K.sum(loss_))
    return tf.cond(tf.math.greater(K.sum(loss_), 0.), lambda: K.sum(loss_), lambda: -K.sum(I))



visible = Input(shape=(79,))
hidden1 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(visible)
hidden2 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden1)
hidden3 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden2)
hidden4 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden3)
hidden5 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden4)
hidden6 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden5)
hidden7 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden6)
output1 = Dense(8, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output2 = Dense(8, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output3 = Dense(8, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output4 = Dense(8, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output5 = Dense(8, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output6 = Dense(8, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output7 = Dense(8, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output8 = Dense(8, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output9 = Dense(79, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output=tf.concat([output1, output2,output3,output4,output5, output6,output7,output8,output9], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.0001, momentum=0.8)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)

#input_=pd.read_table('coeff_bilocality_1.txt', delimiter=" ", header=None)
#input_=input_.to_numpy()
#input_=input_.astype('float32')
input_=np.random.uniform(size=((200000,79)), low=-0.001, high=0.001)
train=tf.constant(input_)

model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0, callbacks=LRchanger())

model.save('bilocality_lib')

