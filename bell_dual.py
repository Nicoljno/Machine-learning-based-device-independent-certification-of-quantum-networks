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

#file=open('FitWeights_opt.txt',"w")
#sys.stdout=file


def custom_loss(train, y_pred):
#    print(y_true.shape, y_pred.shape, y_true.dtype, y_pred.dtype)
    Id=np.eye((5))
    #inizializzazione matrici
    #elementi non fisici
    A0A1=np.zeros((5,5))
    B0B1=np.zeros((5,5))
    A0=np.zeros((5,5))
    A1=np.zeros((5,5))
    B0=np.zeros((5,5))
    B1=np.zeros((5,5))
    A0B0=np.zeros((5,5))
    A0B1=np.zeros((5,5))
    A1B0=np.zeros((5,5))
    A1B1=np.zeros((5,5))
    A0A1[2,1]=A0A1[1,2]=1.
    B0B1[4,3]=B0B1[3,4]=1.
    A0[1,0]=A0[0,1]=1.
    A1[2,0]=A1[0,2]=1.
    B0[3,0]=B0[0,3]=1.
    B1[4,0]=B1[0,4]=1.    
    A0B0[3,1]=A0B0[1,3]=1.
    A0B1[4,1]=A0B1[1,4]=1.
    A1B0[3,2]=A1B0[2,3]=1.
    A1B1[4,2]=A1B1[2,4]=1.
    loss_=[]
    CHSH=[]
    C=np.zeros((5,5))
    C[3,1]=C[1,3]=C[4,1]=C[1,4]=C[3,2]=C[2,3]=1.
    C[4,2]=C[2,4]=-1.
    vals=[0,1]
    p=np.ones((16))
    p=p/4.
    #p=([.5,.5,.5,0,0,0,0,.5,0,0,0,.5,.5,.5,.5,0])
    #p_=np.copy(p)
    #for a in vals:
    #    for b in vals:
    #        for x in vals:
    #            for y in vals:
    #                p[x*8+y*4+a*2+b]=p_[a*8+b*4+x*2+y]
    #p=np.array([0.09982791, 0.5084971 , 0.3482153 , 0.04345965, 0.05634045,
    #            0.50526524, 0.3442526 , 0.09414175, 0.0485099 , 0.41910988,
    #            0.4357971, 0.09658305, 0.1396484 , 0.06417321, 0.08291478,
    #            0.7132636])
    #p=np.array([0.42677669529663675,
    #            0.07322330470336312,
    #            0.07322330470336312,
    #            0.42677669529663675,
    #            0.4267766952966368,
    #            0.07322330470336308,
    #            0.07322330470336308,
    #            0.4267766952966368,
    #            0.4267766952966368,
    #            0.07322330470336308,
    #            0.07322330470336308,
    #            0.4267766952966368,
    #            0.07322330470336313,
    #            0.42677669529663675,
    #            0.42677669529663687,
    #            0.07322330470336312])

    
    for _ in range(1):
        #value1=sum(y_pred[_][a*2+b]-y_pred[_][a*2+b+4] for a in vals for b in vals)
        #value2=sum(y_pred[_][a*2+b]-y_pred[_][a*2+b+8] for a in vals for b in vals)
        CHSH.append(((y_pred[_][26]+train[_][10])+
                     (y_pred[_][27]+train[_][11])+
                     (y_pred[_][28]+train[_][12])+
                     (y_pred[_][29]+train[_][13])+
                     (y_pred[_][30]+train[_][14]))
        tmp=0.5*C-(np.diag([1,0,0,0,0])*(y_pred[_][26]+train[_][10])+
                   np.diag([0,1,0,0,0])*(y_pred[_][27]+train[_][11])+
                   np.diag([0,0,1,0,0])*(y_pred[_][28]+train[_][12])+
                   np.diag([0,0,0,1,0])*(y_pred[_][29]+train[_][13])+
                   np.diag([0,0,0,0,1])*(y_pred[_][30]+train[_][14]))
        loss_.append(-reduce_min(tf.linalg.eigvalsh(tmp)))
    loss_=tf.stack(loss_)
    tf.print(tf.math.reduce_min(CHSH), K.sum(loss_))#, output_stream="file://FitWeights_opt.txt")
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

@tf.RegisterGradient("HeavisideGrad")
def _heaviside_grad(unused_op: tf.Operation, grad: tf.Tensor):
    x = unused_op.inputs[0]
    # During backpropagation heaviside behaves like sigmoid
    return tf.sigmoid(x) * (1 - tf.sigmoid(x)) * grad


def heaviside(x: tf.Tensor, g: tf.Graph = tf.compat.v1.get_default_graph()):
    custom_grads = {
        "Sign": "HeavisideGrad"
    }
    with g.gradient_override_map(custom_grads):
        # heaviside(0) currently returns 0. 
        sign = tf.sign(x)
        step_func = sign + tf.stop_gradient(tf.maximum(0.0, sign) - sign)
        return step_func

@tf.function
@tf.autograph.experimental.do_not_convert
def activation_f(logits):
    NS=0
    vals=[0,1]
    #print(logits.shape)
    out1=logits[:,0:4]
    out2=logits[:,4:8]
    out3=logits[:,8:12]
    out4=logits[:,12:]
    #print(out1.shape, out2.shape, out3.shape, out4.shape)
    out1 = tf.exp(out1) / tf.reduce_sum(tf.exp(out1))
    out2 = tf.exp(out2) / tf.reduce_sum(tf.exp(out2))
    out3 = tf.exp(out3) / tf.reduce_sum(tf.exp(out3))
    out4 = tf.exp(out4) / tf.reduce_sum(tf.exp(out4))
    out=tf.concat([out1, out2, out3, out4], -1)
    #for b in vals:
    #    for y in vals:
    #        NS=NS+abs(sum(out[a*2+b+y*4]-out[a*2+b+8+y*4] for a in vals))
    #for a in vals:
    #    for x in vals:
    #        NS=NS+abs(sum(out[a*2+b+x*8]-out[a*2+b+x*8+4] for a in vals))
    return  out#*(1-heaviside(NS))



#file=open('Samples_pol,txt',"w")
#initializer = tf.keras.initializers.Constant(value=0.3)

visible = Input(shape=(15,))
hidden1 = Dense(260, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(visible)
hidden2 = Dense(260, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden1)
hidden3 = Dense(260, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden2)
hidden4 = Dense(260, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden3)
hidden5 = Dense(260, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden4)
hidden6 = Dense(260, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden5)
hidden7 = Dense(260, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden6)
output1 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output2 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output3 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output4 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
#output1 = Dense(16, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output5 = Dense(15, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output=tf.concat([output1,output2,output3,output4,output5], axis=-1)
opt=keras.optimizers.Adamax(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
#opt=keras.optimizers.SGD(learning_rate=0.0001, momentum=0.1)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)


#input_=pd.read_table('UV.txt', delimiter=" ", header=None)
#input_=input_.to_numpy()
#input_=input_.astype('float32')
#input_=np.concatenate((input_, np.zeros((100,2))), axis=0)
callback=early_stop()
#train=tf.constant(input_)

input_=np.zeros((200000,15))
train=tf.constant(input_)

model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0, callbacks=[callback])

#model.fit(train, train, batch_size=20, epochs=4, shuffle=True, verbose=0, callbacks=[callback, LRchanger()])


model.save('dual_bell')
