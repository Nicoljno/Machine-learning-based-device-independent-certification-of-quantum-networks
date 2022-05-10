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
    Gamma=np.array([[1,0,0,0,0],
                    [0,1,0,0,0],
                    [0,0,1,0,0],
                    [0,0,0,1,0],
                    [0,0,0,0,1]], dtype=np.float32)
    Gamma0=np.array([[0,1,0,0,0],
                    [1,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0]])
    Gamma1=np.array([[0,0,1,0,0],
                    [0,0,0,0,0],
                    [1,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0]])
    Gamma2=np.array([[0,0,0,1,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [1,0,0,0,0],
                    [0,0,0,0,0]])
    Gamma3=np.array([[0,0,0,0,1],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [1,0,0,0,0]])
    Gamma4=np.array([[0,0,0,0,0],
                    [0,0,0,1,0],
                    [0,0,0,0,0],
                    [0,1,0,0,0],
                    [0,0,0,0,0]])
    Gamma5=np.array([[0,0,0,0,0],
                    [0,0,0,0,1],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,1,0,0,0]])
    Gamma6=np.array([[0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,1,0],
                    [0,0,1,0,0],
                    [0,0,0,0,0]])
    Gamma7=np.array([[0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,1],
                    [0,0,0,0,0],
                    [0,0,1,0,0]])
    TMP1=np.array([[0,0,0,0,0],
                   [0,0,1,0,0],
                   [0,1,0,0,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0]])
    TMP2=np.array([[0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,0,0,0,1],
                   [0,0,0,1,0]])
    loss_=[]
    CHSH=[]
    vals=[0,1]
    for _ in range(1):
        CHSH.append(sum((-1)**(a+b)*y_pred[_][a*2+b] for a in vals for b in vals)+
                    sum((-1)**(a+b)*y_pred[_][a*2+b+4] for a in vals for b in vals)+
                    sum((-1)**(a+b)*y_pred[_][a*2+b+8] for a in vals for b in vals)-
                    sum((-1)**(a+b)*y_pred[_][a*2+b+12] for a in vals for b in vals))
        tmp=(TMP1*(train[_][0]+y_pred[_][16])+TMP2*(train[_][1]+y_pred[_][17])+
             Gamma+
             Gamma0*sum((-1)**a*y_pred[_][a*2+b] for a in vals for b in vals)+
             Gamma1*sum((-1)**a*y_pred[_][a*2+b+8] for a in vals for b in vals)+
             Gamma2*sum((-1)**b*y_pred[_][a*2+b] for a in vals for b in vals)+
             Gamma3*sum((-1)**b*y_pred[_][a*2+b+4] for a in vals for b in vals)+
             Gamma4*sum((-1)**(a+b)*y_pred[_][a*2+b] for a in vals for b in vals)+
             Gamma5*sum((-1)**(a+b)*y_pred[_][a*2+b+4] for a in vals for b in vals)+
             Gamma6*sum((-1)**(a+b)*y_pred[_][a*2+b+8] for a in vals for b in vals)+
             Gamma7*sum((-1)**(a+b)*y_pred[_][a*2+b+12] for a in vals for b in vals))
        loss_.append(-reduce_min(tf.linalg.eigvalsh(tmp)))
    loss_=tf.stack(loss_)
    tf.print(tf.math.reduce_min(CHSH), K.sum(loss_))#, output_stream="file://FitWeights_opt.txt")
    return tf.cond(tf.math.greater(K.sum(loss_), 0), lambda: K.sum(loss_), lambda: K.sum(loss_)+K.sum(CHSH))

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

visible = Input(shape=(2,))
hidden1 = Dense(100, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(visible)
hidden2 = Dense(100, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden1)
hidden3 = Dense(100, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden2)
hidden4 = Dense(100, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden3)
hidden5 = Dense(100, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden4)
hidden6 = Dense(100, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden5)
hidden7 = Dense(100, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden6)
output1 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output2 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output3 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output4 = Dense(4, activation = 'softmax', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output5 = Dense(2, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output=tf.concat([output1, output2,output3,output4, output5], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.00001, momentum=0.8)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)


#input_=pd.read_table('UV.txt', delimiter=" ", header=None)
#input_=input_.to_numpy()
#input_=input_.astype('float32')
#input_=np.concatenate((input_, np.zeros((100,2))), axis=0)
callback=early_stop()
#train=tf.constant(input_)

input_=np.zeros((60000,2))
train=tf.constant(input_)

model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0, callbacks=[callback])

#model.fit(train, train, batch_size=20, epochs=4, shuffle=True, verbose=0, callbacks=[callback, LRchanger()])


model.save('test2')
