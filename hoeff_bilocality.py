import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from math import *
from tensorflow import keras
from tensorflow import concat
from tensorflow.keras.callbacks import LambdaCallback, Callback
from tensorflow.math import multiply, reduce_sum, reduce_min, reduce_max
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

def causal_indep_prod(p, y_pred, operator, parties, inputs, mes_diz, non_mes_diz, indep_cons_diz, non_mes_element):
    prod=list(operator.split('*'))
    tmp=1.
    for j in prod:
        if j in mes_diz.keys():
            tmp*=expected_value(p, j, parties, inputs)
        elif j in non_mes_diz.keys():
            tmp*=y_pred[-(64+non_mes_element[j])]
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
    #Azuma errors
    #errors=np.ones(64)
    #errors=errors*0.0356

    #Hoeffding's errors
    errors=np.array([0.01604201, 0.01604201, 0.01604201, 0.01604201, 0.01604201, 0.01604201, 0.01604201, 0.01604201,
                    0.01121466, 0.01121466, 0.01121466, 0.01121466, 0.01121466, 0.01121466, 0.01121466, 0.01121466,
                    0.01103106, 0.01103106, 0.01103106, 0.01103106, 0.01103106, 0.01103106, 0.01103106, 0.01103106,
                    0.01180538, 0.01180538, 0.01180538, 0.01180538, 0.01180538, 0.01180538, 0.01180538, 0.01180538,
                    0.01568708, 0.01568708, 0.01568708, 0.01568708, 0.01568708, 0.01568708, 0.01568708, 0.01568708,
                    0.01109132, 0.01109132, 0.01109132, 0.01109132, 0.01109132, 0.01109132, 0.01109132, 0.01109132,
                    0.01422704, 0.01422704, 0.01422704, 0.01422704, 0.01422704, 0.01422704, 0.01422704, 0.01422704,
                    0.01601731, 0.01601731, 0.01601731, 0.01601731, 0.01601731, 0.01601731, 0.01601731, 0.01601731])
    errors=errors*sqrt(-log(10**(-5)*0.5))/sqrt(-log(10**(-5)*0.5))

    p=train[0][0:64]
    p=K.abs(p+tf.multiply(y_pred[0][-64:], errors.T))
    P1=p[0:8]/K.sum(p[0:8])
    P2=p[8:16]/K.sum(p[8:16])
    P3=p[16:24]/K.sum(p[16:24])
    P4=p[24:32]/K.sum(p[24:32])
    P5=p[32:40]/K.sum(p[32:40])
    P6=p[40:48]/K.sum(p[40:48])
    P7=p[48:56]/K.sum(p[48:56])
    P8=p[56:64]/K.sum(p[56:64])
    p=tf.concat([P1,P2,P3,P4,P5,P6,P7,P8], axis=0)

    loss_=[]
    I=[]
    vals=[0,1]
    keys=list(non_mes_diz.keys())
    for _ in range(1):
        I1=0.25*sum((-1)**(a+b+c)*p[x*32+8*y+a*4+b*2+c] for x in vals for y in vals for a in vals for b in vals for c in vals)
        I2=0.25*sum((-1)**(a+b+c+x+y)*p[x*32+16+y*8+a*4+b*2+c] for x in vals for y in vals for a in vals for b in vals for c in vals)
        I.append(K.sqrt(K.abs(I1))+K.sqrt(K.abs(I2)))
        gamma=(sum(mes_diz[i]*expected_value(p, i, parties, inputs) for i in mes_diz.keys() )+
               sum(non_mes_diz[keys[i]]*(y_pred[_][-(64+i)]) for i in range(len(keys)))+
               sum(causal_indep_prod(p, y_pred[0], i, parties, inputs, mes_diz, non_mes_diz, indep_cons_diz, non_mes_element)*indep_cons_diz[i]
                   for i in indep_cons_diz.keys())
               )
        loss_.append(-reduce_min(tf.linalg.eigvalsh(gamma)))
    no_sign=0
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    no_sign+=abs(sum(p[a*4+b*2+c+x*32+y*16+8]-p[a*4+b*2+c+x*32+y*16] for c in vals))
    for a in vals:
        for c in vals:
            for x in vals:
                for z in vals:
                    no_sign+=abs(sum(p[a*4+b*2+c+x*32+16+z*8]-p[a*4+b*2+c+x*32+z*8] for b in vals))
    for b in vals:
        for c in vals:
            for y in vals:
                for z in vals:
                    no_sign+=abs(sum(p[a*4+b*2+c+32+y*16+z*8]-p[a*4+b*2+c+y*16+z*8] for a in vals))
    tf.print(K.sum(I), K.sum(loss_), no_sign)
    return tf.cond(tf.math.greater(no_sign, 0.1), lambda: no_sign, lambda: tf.cond(tf.math.greater(K.sum(loss_), 0), lambda: K.sum(loss_), lambda: K.sum(I)))


visible = Input(shape=(64,))
hidden1 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(visible)
hidden2 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden1)
hidden3 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden2)
hidden4 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden3)
hidden5 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden4)
hidden6 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden5)
hidden7 = Dense(560, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden6)
output1 = Dense(79, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output2 = Dense(64, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output = tf.concat([output1, output2], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.00001, momentum=0.8)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)

input_=pd.read_table('bilocal_samples.txt', delimiter=" ", header=None)
input_=input_.to_numpy()
input_=input_.astype('float32')
train=tf.constant(input_)
model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0)

model.save('hoeff_bilocality_exp_6')
