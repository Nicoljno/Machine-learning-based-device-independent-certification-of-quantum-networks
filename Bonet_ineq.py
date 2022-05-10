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
import sys


def custom_loss(train, y_pred):
    Gamma=np.diag([1,1,1,1,1,1])
    Gamma0=np.array([[0,0,0,0,0,0],
                     [0,0,1,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]])
    Gamma1=np.array([[0,0,0,0,0,0],
                     [0,0,0,1,0,0],
                     [0,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]])
    Gamma2=np.array([[0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,1,0,0],
                     [0,0,1,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]])
    Gamma3=np.array([[0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,1],
                     [0,0,0,0,1,0]])
    Gamma6=np.array([[0,1,0,0,0,0],
                     [1,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]])
    Gamma7=np.array([[0,0,1,0,0,0],
                     [0,0,0,0,0,0],
                     [1,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]])
    Gamma8=np.array([[0,0,0,1,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [1,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]])
    Gamma9=np.array([[0,0,0,0,1,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [1,0,0,0,0,0],
                     [0,0,0,0,0,0]])
    Gamma10=np.array([[0,0,0,0,0,1],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [1,0,0,0,0,0]])
    Gamma12=np.array([[0,0,0,0,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,1,0,0,0,0],
                      [0,0,0,0,0,0]])
    Gamma13=np.array([[0,0,0,0,0,0],
                      [0,0,0,0,0,1],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,1,0,0,0,0]])
    Gamma15=np.array([[0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,0],
                      [0,0,1,0,0,0],
                      [0,0,0,0,0,0]])
    Gamma16=np.array([[0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,1],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,1,0,0,0]])
    Gamma18=np.array([[0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,1,0,0],
                      [0,0,0,0,0,0]])
    Gamma19=np.array([[0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,1],
                      [0,0,0,0,0,0],
                      [0,0,0,1,0,0]])
    loss_=[]
    BONET=[]
    CHSH=[]
    vals=[0,1]
    for _ in range(1):
        CHSH.append(sum((-1)**(a+b)*y_pred[_][a*2+b] for a in vals for b in vals)+
                    sum((-1)**(a+b)*y_pred[_][a*2+b+4] for a in vals for b in vals)+
                    sum((-1)**(a+b)*y_pred[_][a*2+b+8] for a in vals for b in vals)-
                    sum((-1)**(a+b)*y_pred[_][a*2+b+12] for a in vals for b in vals))        
        BONET.append(y_pred[_][15]+y_pred[_][16]+y_pred[_][1]+y_pred[_][9]+y_pred[_][6])
        tmp=(Gamma0*(train[_][0]+y_pred[_][20])+Gamma1*(train[_][1]+y_pred[_][21])+
             Gamma2*(train[_][2]+y_pred[_][22])+Gamma3*(train[_][3]+y_pred[_][23])+
             Gamma+
             Gamma6*(-y_pred[_][3]+y_pred[_][0]-y_pred[_][2]+y_pred[_][1])+
             Gamma7*(-y_pred[_][11]+y_pred[_][8]-y_pred[_][10]+y_pred[_][9])+
             Gamma8*(y_pred[_][16]+y_pred[_][17]-y_pred[_][18]-y_pred[_][19])+
             Gamma9*(-y_pred[_][3]+y_pred[_][0]+y_pred[_][2]-y_pred[_][1])+
             Gamma10*(-y_pred[_][7]+y_pred[_][4]+y_pred[_][6]-y_pred[_][5])+
             Gamma12*(y_pred[_][3]+y_pred[_][0]-y_pred[_][2]-y_pred[_][1])+
             Gamma13*(y_pred[_][7]+y_pred[_][4]-y_pred[_][6]-y_pred[_][5])+
             Gamma15*(y_pred[_][11]+y_pred[_][8]-y_pred[_][10]-y_pred[_][9])+
             Gamma16*(y_pred[_][15]+y_pred[_][12]-y_pred[_][14]-y_pred[_][13])+
             Gamma18*(y_pred[_][16]-y_pred[_][17])+ #+y_pred[_][19]-y_pred[_][18]
             Gamma19*(y_pred[_][18]-y_pred[_][19])) #+y_pred[_][23]-y_pred[_][22]
        loss_.append(-reduce_min(tf.linalg.eigvalsh(tmp)))
    loss_=tf.stack(loss_)
    tf.print(K.sum(CHSH), K.sum(loss_))
    return tf.cond(tf.math.greater(K.sum(loss_), 0), lambda: K.sum(loss_), lambda: K.sum(loss_)-K.sum(CHSH))

class LRchanger(Callback):
    def __init__(self,display=500):
        self.seen = 0
        self.display = display
    def on_batch_end(self,batch,logs={}):
        self.seen += 1
        #print(self.model.optimizer.lr)
        if self.seen % self.display == 0:
            self.model.optimizer.lr=self.model.optimizer.lr*0.99
            #print(self.model.optimizer.lr)

class early_stop(Callback):
    def __init__(self,display=500):
        self.seen = 0
        self.display = display
    def on_batch_begin(self, batch, logs={}):
        self.seen += 1
        if self.seen % self.display == 0:
            l0=self.model.loss
        def on_batch_end(self,batch,logs={}):
            l1=self.model.loss
            if(l0-l1<0.000001):
                self.model.stop_training = True

print_weights = LambdaCallback(on_batch_end=lambda batch, logs: print(logs["loss"]))

visible = Input(shape=(4,))
hidden1 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(visible)
hidden2 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden1)
hidden3 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden2)
hidden4 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden3)
hidden5 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden4)
hidden6 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden5)
hidden7 = Dense(120, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden6)
output1 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output2 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output3 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output4 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output5 = Dense(2, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output6 = Dense(2, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output7 = Dense(4, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output=tf.concat([output1, output2, output3, output4, output5, output6, output7], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.00001, momentum=0.8)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)


#input_=pd.read_table('coeff_bonet_2.txt', delimiter=" ", header=None)
#input_=input_.to_numpy()
#input_=input_.astype('float32')
#input_=np.concatenate((input_, np.zeros((100,4))), axis=0)
input_=np.zeros((60000,4))
callback=early_stop()
train=tf.constant(input_)

model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0, callbacks=[callback])

#model.fit(train, train, batch_size=20, epochs=4, shuffle=True, verbose=0, callbacks=[callback, LRchanger()])


model.save('Bonet_net')
