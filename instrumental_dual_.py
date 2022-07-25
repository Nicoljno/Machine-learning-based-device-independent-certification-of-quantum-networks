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
    filename = f'matrici_mes_constraints_instrumental_lev_{level}'
    with open(filename, 'rb') as f:
        mes_diz = pickle.load(f)
    filename = f'matrici_non_mes_constraints_instrumental_lev_{level}'
    with open(filename, 'rb') as g:
        non_mes_diz = pickle.load(g)
    filename = f'dual_constraints_instrumental'
    with open(filename, 'rb') as f:
        dual_diz = pickle.load(f)
    filename =  f'dual_labels_instrumental'
    with open(filename, 'rb') as g:
        labels = pickle.load(g)
    dual_keys={}
    for i, j in enumerate(list(dual_diz.keys())):
        dual_keys[j]=i
    parties = ['A','B']
    inputs = ['0','1','2']
    loss_=[]
    lenght=len(mes_diz['A0'][0])
    eye=np.eye(lenght)
    non_mes_keys={}
    j=0
    for i in list(non_mes_diz.keys()):
        non_mes_keys[i]=j
        j+=1
    vals=[0,1]
    dual_elements=len(dual_diz.keys())
    C=(mes_diz['A0B0']/np.sum(mes_diz['A0B0'])+
       mes_diz['A0B1']/np.sum(mes_diz['A0B1'])+
       mes_diz['A1B0']/np.sum(mes_diz['A1B0'])-
       mes_diz['A1B1']/np.sum(mes_diz['A1B1']))
    somma=0
    dual_keys=list(dual_diz.keys())
    somma=np.zeros((lenght,lenght))
    k=0
    lenght=len(mes_diz['A0'][0])
    eye=np.eye(lenght)
    vals=[0,1]
    p=y_pred[0][-20:]
    zeros=np.array([0,0])
    p=tf.concat([p,zeros,zeros], axis=0)
    objective=0
    somma=np.zeros((lenght,lenght))
    k=0
    for _ in range(1):
        for key in mes_diz.keys():
            tmp=np.zeros((lenght,lenght))
            triu=np.triu(mes_diz[key])
            lim=int(np.sum(triu))-1
            tmp2=0
            tmp[np.where(triu==1)[0][0]][np.where(triu==1)[1][0]]=1
            tmp*=sum(y_pred[_][int(k)+j]+train[_][int(k)+j] for j in range(lim))
            tmp2+=tmp
            for i in range(len(np.where(triu==1)[1])-1):
                tmp=np.zeros((lenght,lenght))
                tmp[np.where(triu==1)[0][i+1]][np.where(triu==1)[1][i+1]]=1
                tmp*=-y_pred[_][int(k+i)]+train[_][int(k+i)]
                tmp2+=tmp
            somma+=tmp2
            k+=lim
            objective+=tf.math.reduce_sum(tmp2+tf.transpose(tmp2))*expected_value(p, key, parties, inputs)
        for key in non_mes_diz.keys():
            tmp=np.zeros((lenght,lenght))
            triu=np.triu(non_mes_diz[key])
            lim=int(np.sum(triu))-1
            tmp2=0
            tmp[np.where(triu==1)[0][0]][np.where(triu==1)[1][0]]=1
            tmp*=sum(y_pred[_][int(k)+j]+train[_][int(k)+j] for j in range(lim))
            tmp2+=tmp
            for i in range(len(np.where(triu==1)[1])-1):
                tmp=np.zeros((lenght,lenght))
                tmp[np.where(triu==1)[0][i+1]][np.where(triu==1)[1][i+1]]=1
                tmp*=-y_pred[_][int(k+i)]+train[_][int(k+i)]
                tmp2+=tmp
            somma+=tmp2
            k+=lim
            objective+=tf.cast(tf.math.reduce_sum(tmp2+tf.transpose(tmp2)), tf.float32)*(y_pred[_][-(non_mes_keys[key]+1)]+train[_][-(non_mes_keys[key]+1)])
        somma=somma+tf.transpose(somma)
        gamma=C-somma-sum(np.diag(eye[i])*(y_pred[_][int(k)+i]+train[_][int(k)+i]) for i in range(lenght))
        #k+=lenght
        objective+=sum(y_pred[_][int(k)+i]+train[_][int(k)+i] for i in range(lenght))
        loss_.append(-reduce_min(tf.linalg.eigvalsh(gamma)))
    tf.print(objective, K.sum(loss_))
    no_sign=0
    for b in vals:
       for y in vals:
            no_sign+=abs(sum(p[a*2+b+y*4]-p[a*2+b+8+y*4] for a in vals))
    for a in vals:
        for x in vals:
            no_sign+=abs(sum(p[a*2+b+x*8]-p[a*2+b+x*8+4] for b in vals))
    return tf.cond( tf.math.greater(no_sign, 0.01), lambda: no_sign, lambda: tf.cond(tf.math.greater(K.sum(loss_), 0.), lambda: K.sum(loss_), lambda: -objective))
    #return tf.cond(tf.math.greater(K.sum(loss_), 0.), lambda: K.sum(loss_), lambda: -objective)





class LRchanger(Callback):
    def __init__(self,display=3000):
        self.seen = 0
        self.display = display
    def on_batch_end(self,batch,logs={}):
        self.seen += 1
        #print(self.model.optimizer.lr)
        if self.seen % self.display == 0:
            self.model.optimizer.lr=self.model.optimizer.lr*0.1
            #print(self.model.optimizer.lr)

level=2
filename =  f'matrici_non_mes_constraints_instrumental_lev_{level}'
with open(filename, 'rb') as f:
    non_mes_diz = pickle.load(f)
filename = f'matrici_mes_constraints_instrumental_lev_{level}'
with open(filename, 'rb') as f:
    mes_diz = pickle.load(f)
lenght=len(mes_diz['A0'][0])
N=int((lenght*lenght+lenght)/2+1)+len(non_mes_diz.keys())+len(non_mes_diz.keys())
visible = Input(shape=(N,))
hidden1 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(visible)
hidden2 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden1)
hidden3 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden2)
hidden4 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden3)
hidden5 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden4)
hidden6 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden5)
hidden7 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden6)
output0 = Dense(N, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output1 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output2 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output3 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output4 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output5 = Dense(2, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output6 = Dense(2, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output=tf.concat([output0, output1, output2, output3, output4, output5, output6], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.0001, momentum=0.8)
#opt=keras.optimizers.Adamax(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)

input_=np.zeros((10000,N))
train=tf.constant(input_)

model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0, callbacks=[LRchanger()])

parties = ['A','B']
level = 2

model.save(f'dual_instrumental_{len(parties)}_lev_{level}')
