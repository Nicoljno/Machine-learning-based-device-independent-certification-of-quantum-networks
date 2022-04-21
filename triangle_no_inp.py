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

'''
def custom_loss(train, y_pred):
    Id=np.eye(7)
    A=np.zeros((7,7))
    B=np.zeros((7,7))
    C=np.zeros((7,7))
    AB=np.zeros((7,7))
    AC=np.zeros((7,7))
    BC=np.zeros((7,7))
    ABC=np.zeros((7,7))
    A[0,1]=A[1,0]=A[4,2]=A[2,4]=A[5,3]=A[3,5]=1.
    B[0,2]=B[2,0]=B[4,1]=B[1,4]=B[6,3]=B[3,6]=1.
    C[0,3]=C[3,0]=C[5,1]=C[1,5]=C[6,2]=C[2,6]=1.
    AB[2,1]=AB[1,2]=AB[4,0]=AB[0,4]=AB[6,5]=AB[5,6]=1.
    AC[5,0]=AC[0,5]=AC[3,1]=AC[1,3]=AC[6,4]=AC[4,6]=1.
    BC[6,0]=BC[0,6]=BC[3,2]=BC[2,3]=BC[5,4]=BC[4,5]=1.
    ABC[6,1]=ABC[1,6]=ABC[4,3]=ABC[3,4]=ABC[5,2]=ABC[2,5]=1.
    loss_=[]
    I=[]
    vals=[0,1]
    for _ in range(1):
        I.append(sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
                 sum((-1)**(b+c)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)-
                 sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)*sum((-1)**b*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals))
        Gamma=(Id+ABC*(y_pred[_][8]+train[_][0])+
               A*sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B*sum((-1)**b*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               C*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               AB*sum((-1)**(a+b)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               AC*sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               BC*sum((-1)**(b+c)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals))
        loss_.append(-reduce_min(tf.linalg.eigvalsh(Gamma)))
    tf.print(K.sum(I), K.sum(loss_))
    #return K.sum(loss_)
    return tf.cond(tf.math.greater(K.sum(loss_), 0.), lambda: K.sum(loss_), lambda: -K.sum(I))



visible = Input(shape=(1,))
hidden1 = Dense(56, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(visible)
hidden2 = Dense(56, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden1)
hidden3 = Dense(56, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden2)
hidden4 = Dense(56, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden3)
hidden5 = Dense(56, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden4)
hidden6 = Dense(56, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden5)
hidden7 = Dense(56, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden6)
output = Dense(8, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output1 = Dense(1, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output=tf.concat([output, output1], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.0005, momentum=0.8)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)


#input_=np.random.uniform(size=20000, low=-0.1, high=0.1)
#input_=np.concatenate((input_, np.zeros((1000,57))), axis=0)
input_=np.zeros((20000,1))
train=tf.constant(input_)

model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0)

model.save('triangle_no_inputs')
'''




def custom_loss(train, y_pred):
    Id=np.eye(8)
    A=np.zeros((8,8))
    B=np.zeros((8,8))
    C=np.zeros((8,8))
    AB=np.zeros((8,8))
    AC=np.zeros((8,8))
    BC=np.zeros((8,8))
    ABC=np.zeros((8,8))
    A[0,1]=A[1,0]=A[4,2]=A[2,4]=A[5,3]=A[3,5]=A[7,6]=A[6,7]=1.
    B[0,2]=B[2,0]=B[4,1]=B[1,4]=B[6,3]=B[3,6]=B[7,5]=B[5,7]=1.
    C[0,3]=C[3,0]=C[5,1]=C[1,5]=C[6,2]=C[2,6]=C[7,4]=C[4,7]=1.
    AB[2,1]=AB[1,2]=AB[4,0]=AB[0,4]=AB[6,5]=AB[5,6]=AB[7,3]=AB[3,7]=1.
    AC[5,0]=AC[0,5]=AC[3,1]=AC[1,3]=AC[6,4]=AC[4,6]=AC[7,2]=AC[2,7]=1.
    BC[6,0]=BC[0,6]=BC[3,2]=BC[2,3]=BC[5,4]=BC[4,5]=BC[7,1]=BC[1,7]=1.
    ABC[6,1]=ABC[1,6]=ABC[4,3]=ABC[3,4]=ABC[5,2]=ABC[2,5]=ABC[7,0]=ABC[0,7]=1.
    loss_=[]
    I=[]
    vals=[0,1]
    for _ in range(1):
        I.append(sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
                 sum((-1)**(b+c)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)-
                 sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)*sum((-1)**b*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals))
        Gamma=(Id+ABC*(y_pred[_][8]+train[_][0])+
               A*sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B*sum((-1)**b*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               C*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               AB*sum((-1)**(a+b)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               AC*sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               BC*sum((-1)**(b+c)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals))
        loss_.append(-reduce_min(tf.linalg.eigvalsh(Gamma)))
    tf.print(K.sum(I), K.sum(loss_))
    #return K.sum(loss_)
    return tf.cond(tf.math.greater(K.sum(loss_), 0.), lambda: K.sum(loss_), lambda: -K.sum(I))



visible = Input(shape=(1,))
hidden1 = Dense(86, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(visible)
hidden2 = Dense(86, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden1)
hidden3 = Dense(86, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden2)
hidden4 = Dense(86, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden3)
hidden5 = Dense(86, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden4)
hidden6 = Dense(86, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden5)
hidden7 = Dense(86, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden6)
output = Dense(8, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output1 = Dense(1, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output=tf.concat([output, output1], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.0005, momentum=0.8)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)


#input_=np.random.uniform(size=20000, low=-0.1, high=0.1)
#input_=np.concatenate((input_, np.zeros((1000,57))), axis=0)
input_=np.zeros((20000,1))
train=tf.constant(input_)

model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0)

model.save('triangle_no_inputs_3')

