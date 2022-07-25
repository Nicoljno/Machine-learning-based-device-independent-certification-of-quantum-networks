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

#file=open('FitWeights_opt.txt',"w")
#sys.stdout=file

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
    parties = ['A','B']
    level = 2
    filename = 'matrici_mes_constraints_bell'
    with open(filename, 'rb') as f:
        mes_diz = pickle.load(f)
    filename = 'matrici_non_mes_constraints_bell'
    with open(filename, 'rb') as g:
        non_mes_diz = pickle.load(g)
    filename = f'mes_dual_constraints_bell'
    with open(filename, 'rb') as f:
        mes_dual_diz = pickle.load(f)
    filename = f'non_mes_dual_constraints_bell'
    with open(filename, 'rb') as f:
        non_mes_dual_diz = pickle.load(f)
    filename =  f'mes_dual_labels_bell'
    with open(filename, 'rb') as g:
        mes_labels = pickle.load(g)
    filename =  f'non_mes_dual_labels_bell'
    with open(filename, 'rb') as g:
        non_mes_labels = pickle.load(g)
    mes_dual_keys={}
    for i, j in enumerate(list(mes_dual_diz.keys())):
        mes_dual_keys[j]=i
    non_mes_dual_keys={}
    for i, j in enumerate(list(non_mes_dual_diz.keys())):
        non_mes_dual_keys[j]=i
    parties = ['A','B']
    inputs = ['0','1']
    loss_=[]
    lenght=len(mes_diz['A0'][0])
    eye=np.eye(lenght)
    vals=[0,1]
    #dual_elements=len(dual_diz.keys())
    C=(mes_diz['A0B0']/np.sum(mes_diz['A0B0'])+
       mes_diz['A0B1']/np.sum(mes_diz['A0B1'])+
       mes_diz['A1B0']/np.sum(mes_diz['A1B0'])-
       mes_diz['A1B1']/np.sum(mes_diz['A1B1']))
    somma=0
    mes_dual_keys=list(mes_dual_diz.keys())
    non_mes_dual_keys=list(non_mes_dual_diz.keys())
    mes_keys=list(mes_diz.keys())
    somma=np.zeros((lenght,lenght))
    k=0
    m=1
    for _ in range(1):
        for tmp in mes_labels:
            for j in range(k+1,tmp):
                somma+=mes_dual_diz[mes_dual_keys[k]]*(y_pred[_][j+lenght+16]+train[_][j+lenght+16])+mes_dual_diz[mes_dual_keys[j]]*(y_pred[_][j+lenght+16]+train[_][j+lenght+16])
            k=tmp
        k=0
        for tmp in non_mes_labels:
            for j in range(k+1,tmp):
                somma+=non_mes_dual_diz[non_mes_dual_keys[k]]*(y_pred[_][j+16+lenght+len(mes_dual_diz.keys())]+train[_][j+16+lenght+len(mes_dual_diz.keys())])+non_mes_dual_diz[non_mes_dual_keys[j]]*(y_pred[_][j+16+lenght+len(mes_dual_diz.keys())]+train[_][j+16+lenght+len(mes_dual_diz.keys())])
            k=tmp
        objective=sum(y_pred[_][i+16]+train[_][i+16] for i in range(lenght))
        k=0
        for tmp in mes_labels:
            for j in range(k+1,tmp):
                objective+=expected_value(y_pred[_][0:16], mes_keys[m], parties, inputs)*(y_pred[_][j+16+lenght]+train[_][j+lenght+16])/(tmp-k)-expected_value(y_pred[_][0:16], mes_keys[m], parties, inputs)*(y_pred[_][j+lenght+16]+train[_][j+lenght+16])
            m+=1
            k=tmp
        gamma=C-sum(np.diag(eye[i])*(y_pred[_][i+16]+train[_][i+16]) for i in range(lenght))-somma
    no_sign=0
    vals=[0,1]
    for b in vals:
        for y in vals:
            no_sign+=abs(sum(y_pred[0][a*2+b+y*4]-y_pred[0][a*2+b+8+y*4] for a in vals))
    for a in vals:
        for x in vals:
            no_sign+=abs(sum(y_pred[0][a*2+b+x*8]-y_pred[0][a*2+b+x*8+4] for b in vals))
    tf.print(objective, -reduce_min(tf.linalg.eigvalsh(gamma)), no_sign)
    return tf.cond(tf.math.greater(no_sign, 0.001), lambda: no_sign, lambda: tf.cond(tf.math.greater(-reduce_min(tf.linalg.eigvalsh(gamma)), 0.), lambda: -reduce_min(tf.linalg.eigvalsh(gamma)), lambda: -reduce_min(tf.linalg.eigvalsh(gamma))-objective))


visible = Input(shape=(107,))
hidden1 = Dense(300, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(visible)
hidden2 = Dense(300, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden1)
hidden3 = Dense(300, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden2)
hidden4 = Dense(300, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden3)
hidden5 = Dense(300, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden4)
hidden6 = Dense(300, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden5)
hidden7 = Dense(300, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden6)
output1 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output2 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output3 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output4 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output5 = Dense(91, activation = 'tanh')(hidden7)
output=tf.concat([output1, output2,output3,output4, output5], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.00001, momentum=0.8)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)

input_=np.zeros((100000,107))
train=tf.constant(input_)

model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0)

parties = ['A','B']
level = 2

model.save(f'dual_bell_{len(parties)}_lev_{level}')
