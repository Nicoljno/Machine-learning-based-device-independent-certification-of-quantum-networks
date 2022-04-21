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
    Id=np.eye(22)
    #inizializzazione matrici
    #elementi non fisici
    A0A1=np.zeros((22,22))
    B0B1=np.zeros((22,22))
    C0C1=np.zeros((22,22))
    A0A1A0=np.zeros((22,22))
    A1A0A1=np.zeros((22,22))
    B0B1B0=np.zeros((22,22))
    B1B0B1=np.zeros((22,22))
    C0C1C0=np.zeros((22,22))
    C1C0C1=np.zeros((22,22))
    A0B0B1=np.zeros((22,22))
    A1B0B1=np.zeros((22,22))
    A0C0C1=np.zeros((22,22))
    A1C0C1=np.zeros((22,22))
    B0C0C1=np.zeros((22,22))
    B1C0C1=np.zeros((22,22))
    A0A1B0=np.zeros((22,22))
    A0A1B1=np.zeros((22,22))
    A0A1C0=np.zeros((22,22))
    A0A1C1=np.zeros((22,22))
    B0B1C0=np.zeros((22,22))
    B0B1C1 = np.zeros((22,22))
    A0A1A0B0=np.zeros((22,22))
    A0A1A0B1=np.zeros((22,22))
    A0A1A0C0=np.zeros((22,22))
    A0A1A0C1=np.zeros((22,22))
    A1A0A1B0=np.zeros((22,22))
    A1A0A1B1=np.zeros((22,22))
    A1A0A1C0=np.zeros((22,22))
    A1A0A1C1=np.zeros((22,22))
    B0B1B0A0=np.zeros((22,22))
    B0B1B0A1=np.zeros((22,22))
    B0B1B0C0=np.zeros((22,22))
    B0B1B0C1=np.zeros((22,22))
    B1B0B1A0=np.zeros((22,22))
    B1B0B1A1=np.zeros((22,22))
    B1B0B1C0=np.zeros((22,22))
    B1B0B1C1=np.zeros((22,22))
    C0C1C0A0=np.zeros((22,22))
    C0C1C0A1=np.zeros((22,22))
    C0C1C0B0=np.zeros((22,22))
    C0C1C0B1=np.zeros((22,22))
    C1C0C1A0=np.zeros((22,22))
    C1C0C1A1=np.zeros((22,22))
    C1C0C1B0=np.zeros((22,22))
    C1C0C1B1=np.zeros((22,22))
    A0B0C0C1=np.zeros((22,22))
    A0B1C0C1=np.zeros((22,22))
    A1B0C0C1=np.zeros((22,22))
    A1B1C0C1=np.zeros((22,22))
    A0A1B0B1=np.zeros((22,22))
    A0A1C0C1=np.zeros((22,22))
    A0A1B1B0=np.zeros((22,22))
    A0A1C1C0=np.zeros((22,22))
    A0A1B1C0=np.zeros((22,22))
    A0A1B1C1=np.zeros((22,22))
    A0A1B0C0=np.zeros((22,22))
    A0A1B0C1=np.zeros((22,22))
    A0C0B0B1=np.zeros((22,22))
    A0C1B0B1=np.zeros((22,22))
    B0B1C0C1=np.zeros((22,22))
    A1C1B0B1=np.zeros((22,22))
    A1C0B0B1=np.zeros((22,22))
    B0B1C1C0=np.zeros((22,22))

    #elementi fisici
    A0=np.zeros((22,22))
    A1=np.zeros((22,22))
    B0=np.zeros((22,22))
    B1=np.zeros((22,22))
    C0=np.zeros((22,22))
    C1=np.zeros((22,22))
    A0B0=np.zeros((22,22))
    A0B1=np.zeros((22,22))
    A0C0=np.zeros((22,22))
    A0C1=np.zeros((22,22))
    A1B0=np.zeros((22,22))
    A1B1=np.zeros((22,22))
    A1C0=np.zeros((22,22))
    A1C1=np.zeros((22,22))
    B0C0=np.zeros((22,22))
    B0C1=np.zeros((22,22))
    B1C0=np.zeros((22,22))
    B1C1=np.zeros((22,22))
    A0B0C0=np.zeros((22,22))
    A0B1C0=np.zeros((22,22))
    A0B0C1=np.zeros((22,22))
    A0B1C1=np.zeros((22,22))
    A1B0C0=np.zeros((22,22))
    A1B1C0=np.zeros((22,22))
    A1B0C0=np.zeros((22,22))
    A1B0C1=np.zeros((22,22))
    A1B1C1=np.zeros((22,22))

    #riempimento matrici
    #elementi non fisici
    A0A1[2,1]=A0A1[1,2]=A0A1[7,0]=A0A1[0,7]=A0A1[18,8]=A0A1[8,18]=A0A1[19,9]=A0A1[9,19]=A0A1[20,10]=A0A1[10,20]=A0A1[21,11]=A0A1[11,21]=1.
    B0B1[4,3]=B0B1[3,4]=B0B1[12,0]=B0B1[0,12]=B0B1[9,8]=B0B1[8,9]=B0B1[15,13]=B0B1[13,15]=B0B1[16,14]=B0B1[14,16]=B0B1[19,18]=B0B1[18,19]=1.
    C0C1[6,5]=C0C1[5,6]=C0C1[17,0]=C0C1[0,17]=C0C1[11,10]=C0C1[10,11]=C0C1[14,13]=C0C1[13,14]=C0C1[16,15]=C0C1[15,16]=C0C1[21,20]=C0C1[20,21]=1.
    A0A1A0[7,1]=A0A1A0[1,7]=1.
    A1A0A1[7,2]=A1A0A1[2,7]=1.
    B0B1B0[12,3]=B0B1B0[3,12]=1.
    B1B0B1[12,4]=B1B0B1[4,12]=1.
    C0C1C0[17,5]=C0C1C0[5,17]=1.
    C1C0C1[17,6]=C1C0C1[6,17]=1.
    A0B0B1[12,1]=A0B0B1[1,12]=A0B0B1[9,3]=A0B0B1[3,9]=A0B0B1[8,4]=A0B0B1[4,8]=1.
    A1B0B1[12,2]=A1B0B1[2,12]=A1B0B1[19,3]=A1B0B1[3,19]=A1B0B1[18,4]=A1B0B1[4,18]=1.
    A0C0C1[11,5]=A0C0C1[5,11]=A0C0C1[17,1]=A0C0C1[1,17]=A0C0C1[10,6]=A0C0C1[6,10]=1.
    A1C0C1[17,2]=A1C0C1[2,17]=A1C0C1[20,6]=A1C0C1[6,20]=A1C0C1[21,5]=A1C0C1[5,21]=1.
    B0C0C1[13,6]=B0C0C1[6,13]=B0C0C1[14,5]=B0C0C1[5,14]=B0C0C1[17,3]=B0C0C1[3,17]=1.
    B1C0C1[15,6]=B1C0C1[6,15]=B1C0C1[17,4]=B1C0C1[4,17]=B1C0C1[16,5]=B1C0C1[5,16]=1.
    A0A1B0[8,2]=A0A1B0[2,8]=A0A1B0[7,3]=A0A1B0[3,7]=A0A1B0[18,1]=A0A1B0[1,18]=1.
    A0A1B1[9,2]=A0A1B1[2,9]=A0A1B1[7,4]=A0A1B1[4,7]=A0A1B1[19,1]=A0A1B1[1,19]=1.
    A0A1C0[10,2]=A0A1C0[2,10]=A0A1C0[7,5]=A0A1C0[5,7]=A0A1C0[20,1]=A0A1C0[1,20]=1.
    A0A1C1[7,6]=A0A1C1[6,7]=A0A1C1[21,1]=A0A1C1[1,21]=A0A1C1[11,2]=A0A1C1[2,11]=1.
    B0B1C0[12,5]=B0B1C0[5,12]=B0B1C0[13,4]=B0B1C0[4,13]=B0B1C0[15,3]=B0B1C0[3,15]=1.
    B0B1C1[12,6]=B0B1C1[6,12]=B0B1C1[16,3]=B0B1C1[3,16]=B0B1C1[14,4]=B0B1C1[4,14]=1.
    A0A1A0B0[8,7]=A0A1A0B0[7,8]=1.
    A0A1A0B1[9,7]=A0A1A0B1[7,9]=1.
    A0A1A0C0[10,7]=A0A1A0C0[7,10]=1.
    A0A1A0C1[11,7]=A0A1A0C1[7,11]=1.
    A1A0A1B0[18,7]=A1A0A1B0[7,18]=1.
    A1A0A1B1[19,7]=A1A0A1B1[7,19]=1.
    A1A0A1C0[20,7]=A1A0A1C0[7,20]=1.
    A1A0A1C1[21,7]=A1A0A1C1[7,21]=1.
    B0B1B0A0[12,8]=B0B1B0A0[8,12]=1.
    B0B1B0A1[12,18]=B0B1B0A1[18,12]=1.
    B0B1B0C0[12,13]=B0B1B0C0[13,12]=1.
    B0B1B0C1[12,14]=B0B1B0C1[14,12]=1.
    B1B0B1A0[12,9]=B1B0B1A0[9,12]=1.
    B1B0B1A1[12,19]=B1B0B1A1[19,12]=1.
    B1B0B1C0[12,15]=B1B0B1C0[15,12]=1.
    B1B0B1C1[12,16]=B1B0B1C1[16,12]=1.
    C0C1C0A0[17,10]=C0C1C0A0[10,17]=1.
    C0C1C0A1[17,20]=C0C1C0A1[20,17]=1.
    C0C1C0B0[17,13]=C0C1C0B0[13,17]=1.
    C0C1C0B1[17,15]=C0C1C0B1[15,17]=1.
    C1C0C1A0[17,11]=C1C0C1A0[11,17]=1.
    C1C0C1A1[21,17]=C1C0C1A1[17,21]=1.
    C1C0C1B0[17,14]=C1C0C1B0[14,17]=1.
    C1C0C1B1[16,17]=C1C0C1B1[17,16]=1.
    A0B0C0C1[14,10]=A0B0C0C1[10,14]=A0B0C0C1[13,11]=A0B0C0C1[11,13]=A0B0C0C1[17,8]=A0B0C0C1[8,17]=1.
    A0B1C0C1[15,11]=A0B1C0C1[11,15]=A0B1C0C1[16,10]=A0B1C0C1[10,16]=A0B1C0C1[17,9]=A0B1C0C1[9,17]=1.
    A1B0C0C1[18,17]=A1B0C0C1[17,18]=A1B0C0C1[20,14]=A1B0C0C1[14,20]=A1B0C0C1[21,13]=A1B0C0C1[13,21]=1.
    A1B1C0C1[19,17]=A1B1C0C1[17,19]=A1B1C0C1[20,16]=A1B1C0C1[16,20]=A1B1C0C1[21,15]=A1B1C0C1[15,21]=1.
    A0A1B0B1[12,7]=A0A1B0B1[7,12]=A0A1B0B1[19,8]=A0A1B0B1[8,19]=1.
    A0A1C0C1[17,7]=A0A1C0C1[7,17]=A0A1C0C1[21,10]=A0A1C0C1[10,21]=1.
    A0A1B1B0[18,9]=A0A1B1B0[9,18]=1.
    A0A1C1C0[20,11]=A0A1C1C0[11,20]=1.
    A0A1B1C0[19,10]=A0A1B1C0[10,19]=A0A1B1C0[15,7]=A0A1B1C0[7,15]=A0A1B1C0[20,9]=A0A1B1C0[9,20]=1.
    A0A1B1C1[19,11]=A0A1B1C1[11,19]=A0A1B1C1[21,9]=A0A1B1C1[9,21]=A0A1B1C1[16,7]=A0A1B1C1[7,16]=1.
    A0A1B0C0[13,7]=A0A1B0C0[7,13]=A0A1B0C0[18,10]=A0A1B0C0[10,18]=A0A1B0C0[20,8]=A0A1B0C0[8,20]=1.
    A0A1B0C1[18,11]=A0A1B0C1[11,18]=A0A1B0C1[21,8]=A0A1B0C1[8,21]=A0A1B0C1[14,7]=A0A1B0C1[7,14]=1.
    A0C0B0B1[12,10]=A0C0B0B1[10,12]=A0C0B0B1[13,9]=A0C0B0B1[9,13]=A0C0B0B1[15,8]=A0C0B0B1[8,15]=1.
    A0C1B0B1[12,11]=A0C1B0B1[11,12]=A0C1B0B1[14,9]=A0C1B0B1[9,14]=A0C1B0B1[16,8]=A0C1B0B1[8,16]=1.
    B0B1C0C1[16,13]=B0B1C0C1[13,16]=B0B1C0C1[17,12]=B0B1C0C1[12,17]=1.
    A1C1B0B1[18,16]=A1C1B0B1[16,18]=A1C1B0B1[19,14]=A1C1B0B1[14,19]=A1C1B0B1[21,12]=A1C1B0B1[12,21]=1.
    A1C0B0B1[18,15]=A1C0B0B1[15,18]=A1C0B0B1[19,13]=A1C0B0B1[13,19]=A1C0B0B1[20,12]=A1C0B0B1[12,20]=1.
    B0B1C1C0[15,14]=B0B1C1C0[14,15]=1.
    vals=[0,1]
    #elementi fisici   
    A0[1,0]=A0[0,1]=A0[8,3]=A0[3,8]=A0[9,4]=A0[4,9]=A0[10,5]=A0[5,10]=A0[11,6]=A0[6,11]=1.
    A1[2,0]=A1[0,2]=A1[18,3]=A1[3,18]=A1[19,4]=A1[4,19]=A1[20,5]=A1[5,20]=A1[21,6]=A1[6,21]=1.
    B0[3,0]=B0[0,3]=B0[18,2]=B0[2,18]=B0[8,1]=B0[1,8]=B0[14,6]=B0[6,14]=B0[13,5]=B0[5,13]=1.
    B1[4,0]=B1[0,4]=B1[9,1]=B1[1,9]=B1[19,2]=B1[2,19]=B1[15,5]=B1[5,15]=B1[16,6]=B1[6,16]=1.
    C0[5,0]=C0[0,5]=C0[10,1]=C0[1,10]=C0[20,2]=C0[2,20]=C0[13,3]=C0[3,13]=C0[15,4]=C0[4,15]=1.
    C1[6,0]=C1[0,6]=C1[11,1]=C1[1,11]=C1[21,2]=C1[2,21]=C1[14,3]=C1[3,14]=C1[16,4]=C1[4,16]=1.
    A0B0[8,0]=A0B0[0,8]=A0B0[13,10]=A0B0[10,13]=A0B0[14,11]=A0B0[11,14]=A0B0[3,1]=A0B0[1,3]=1.
    A0B1[9,0]=A0B1[0,9]=A0B1[15,10]=A0B1[10,15]=A0B1[16,11]=A0B1[11,16]=A0B1[4,1]=A0B1[1,4]=1.
    A0C0[10,0]=A0C0[0,10]=A0C0[13,8]=A0C0[8,13]=A0C0[15,9]=A0C0[9,15]=A0C0[5,1]=A0C0[1,5]=1.
    A0C1[11,0]=A0C1[0,11]=A0C1[14,8]=A0C1[8,14]=A0C1[16,9]=A0C1[9,16]=A0C1[6,1]=A0C1[1,6]=1.
    A1B0[18,0]=A1B0[0,18]=A1B0[3,2]=A1B0[2,3]=A1B0[20,13]=A1B0[13,20]=A1B0[21,14]=A1B0[14,21]=1.
    A1B1[19,0]=A1B1[0,19]=A1B1[4,2]=A1B1[2,4]=A1B1[20,15]=A1B1[15,20]=A1B1[21,16]=A1B1[16,21]=1.
    A1C0[20,0]=A1C0[0,20]=A1C0[5,2]=A1C0[2,5]=A1C0[18,13]=A1C0[13,18]=A1C0[19,15]=A1C0[15,19]=1.
    A1C1[21,0]=A1C1[0,21]=A1C1[6,2]=A1C1[2,6]=A1C1[18,14]=A1C1[14,18]=A1C1[19,16]=A1C1[16,19]=1.
    B0C0[13,0]=B0C0[0,13]=B0C0[5,3]=B0C0[3,5]=B0C0[10,8]=B0C0[8,10]=B0C0[20,18]=B0C0[18,20]=1.
    B0C1[14,0]=B0C1[0,14]=B0C1[6,3]=B0C1[3,6]=B0C1[11,8]=B0C1[8,11]=B0C1[21,18]=B0C1[18,21]=1.
    B1C0[15,0]=B1C0[0,15]=B1C0[5,4]=B1C0[4,5]=B1C0[10,9]=B1C0[9,10]=B1C0[20,19]=B1C0[19,20]=1.
    B1C1[16,0]=B1C1[0,16]=B1C1[6,4]=B1C1[4,6]=B1C1[11,9]=B1C1[9,11]=B1C1[21,19]=B1C1[19,21]=1.
    A0B0C0[13,1]=A0B0C0[1,13]=A0B0C0[10,3]=A0B0C0[3,10]=A0B0C0[8,5]=A0B0C0[5,8]=1.
    A0B1C0[15,1]=A0B1C0[1,15]=A0B1C0[10,4]=A0B1C0[4,10]=A0B1C0[9,5]=A0B1C0[5,9]=1.
    A0B0C1[14,1]=A0B0C1[1,14]=A0B0C1[11,3]=A0B0C1[3,11]=A0B0C1[8,6]=A0B0C1[6,8]=1.
    A0B1C1[16,1]=A0B1C1[1,16]=A0B1C1[11,4]=A0B1C1[4,11]=A0B1C1[9,6]=A0B1C1[6,9]=1.
    A1B0C0[13,2]=A1B0C0[2,13]=A1B0C0[20,3]=A1B0C0[3,20]=A1B0C0[18,5]=A1B0C0[5,18]=1.
    A1B1C0[15,2]=A1B1C0[2,15]=A1B1C0[20,4]=A1B1C0[4,20]=A1B1C0[19,5]=A1B1C0[5,19]=1.
    A1B0C1[14,2]=A1B0C1[2,14]=A1B0C1[21,3]=A1B0C1[3,21]=A1B0C1[18,6]=A1B0C1[6,18]=1.
    A1B1C1[16,2]=A1B1C1[2,16]=A1B1C1[21,4]=A1B1C1[4,21]=A1B1C1[19,6]=A1B1C1[6,19]=1.
    loss_=[]
    I=[]
    for _ in range(1):
        I1=0.25*sum((-1)**(a+b+c)*y_pred[_][x*32+8*y+a*4+b*2+c] for x in vals for y in vals for a in vals for b in vals for c in vals)
        I2=0.25*sum((-1)**(a+b+c+x+y)*y_pred[_][x*32+16+y*8+a*4+b*2+c] for x in vals for y in vals for a in vals for b in vals for c in vals)
        I.append(K.sqrt(K.abs(I1))+K.sqrt(K.abs(I2)))
        Gamma=(Id+A0A1*(y_pred[_][64]+train[_][0])+B0B1*(y_pred[_][65]+train[_][1])+C0C1*(y_pred[_][66]+train[_][2])+
               A0A1C1C0*((y_pred[_][64]+train[_][0])*(train[_][2]+y_pred[_][66]))+A0A1A0*(y_pred[_][67]+train[_][3])+A1A0A1*(y_pred[_][68]+train[_][4])+B0B1B0*(y_pred[_][69]+train[_][5])+
               B1B0B1*(y_pred[_][70]+train[_][6])+B0B1B0C0*(y_pred[_][71]+train[_][7])+C0C1C0*(y_pred[_][72]+train[_][8])+C1C0C1*(y_pred[_][73]+train[_][9])+A0B0B1*(y_pred[_][74]+train[_][10])+A1B0B1*(y_pred[_][75]+train[_][11])+
               A0C0C1*((y_pred[_][66]+train[_][2])*sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals))+
               A1C0C1*((y_pred[_][66]+train[_][2])*sum((-1)**a*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals))+
               B1B0B1C0*(y_pred[_][76]+train[_][12])+B1B0B1C1*(y_pred[_][77]+train[_][13])+
               B0C0C1*(y_pred[_][78]+train[_][14])+B1C0C1*(y_pred[_][79]+train[_][15])+A0A1B0*(y_pred[_][80]+train[_][16])+A0A1B1*(y_pred[_][81]+train[_][17])+
               A0A1C0*((y_pred[_][64]+train[_][0])*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals))+
               A0A1C1*((y_pred[_][64]+train[_][0])*sum((-1)**c*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals))+
               B0B1C1C0*(y_pred[_][82]+train[_][18])+B0B1C0C1*(y_pred[_][83]+train[_][19])+A0C0B0B1*(y_pred[_][84]+train[_][20])+
               B0B1C0*(y_pred[_][85]+train[_][21])+B0B1C1*(y_pred[_][86]+train[_][22])+A0A1A0B0*(y_pred[_][87]+train[_][23])+A0A1A0B1*(y_pred[_][88]+train[_][24])+
               A0A1A0C0*((y_pred[_][67]+train[_][3])*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals))+
               A0A1A0C1*((y_pred[_][67]+train[_][3])*sum((-1)**c*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals))+
               A1A0A1B0*(y_pred[_][89]+train[_][25])+A1A0A1B1*(y_pred[_][90]+train[_][26])+
               A1A0A1C0*((y_pred[_][68]+train[_][4])*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals))+
               A1A0A1C1*((y_pred[_][68]+train[_][4])*sum((-1)**c*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals))+
               B0B1B0A0*(y_pred[_][91]+train[_][27])+B0B1B0A1*(y_pred[_][92]+train[_][28])+B0B1B0C1*(y_pred[_][93]+train[_][29])+B1B0B1A0*(y_pred[_][94]+train[_][30])+
               B1B0B1A1*(y_pred[_][95]+train[_][31])+C0C1C0B0*(y_pred[_][96]+train[_][32])+C0C1C0B1*(y_pred[_][97]+train[_][33])+
               C0C1C0A0*((y_pred[_][72]+train[_][8])*sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals))+
               C0C1C0A1*((y_pred[_][72]+train[_][8])*sum((-1)**a*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals))+
               C1C0C1A0*((y_pred[_][73]+train[_][9])*sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals))+
               C1C0C1A1*((y_pred[_][73]+train[_][9])*sum((-1)**a*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals))+
               C1C0C1B0*(y_pred[_][98]+train[_][34])+C1C0C1B1*(y_pred[_][99]+train[_][35])+
               A0B0C0C1*(y_pred[_][100]+train[_][36])+A0B1C0C1*(y_pred[_][101]+train[_][37])+
               A1B0C0C1*(y_pred[_][102]+train[_][38])+A1B1C0C1*(y_pred[_][103]+train[_][39])+A0A1B0B1*(y_pred[_][104]+train[_][40])+
               A0A1C0C1*((y_pred[_][64]+train[_][0])*(train[_][2]+y_pred[_][66]))+
               A0A1B1C0*(y_pred[_][105]+train[_][41])+A0A1B0C0*(y_pred[_][106]+train[_][42])+
               A0A1B0C1*(y_pred[_][107]+train[_][43])+A0A1B1B0*(y_pred[_][108]+train[_][44])+A0A1B1C1*(y_pred[_][109]+train[_][45])+
               A0C1B0B1*(y_pred[_][110]+train[_][46])+A1C1B0B1*(y_pred[_][111]+train[_][47])+A1C0B0B1*(y_pred[_][112]+train[_][48])+
               A0*sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1*sum((-1)**a*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B0*sum((-1)**b*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B1*sum((-1)**b*y_pred[_][8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               C0*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               C1*sum((-1)**c*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B0*sum((-1)**(a+b)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B1*sum((-1)**(a+b)*y_pred[_][8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0C0*sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0C1*sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)*sum((-1)**c*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B0*sum((-1)**(a+b)*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B1*sum((-1)**(a+b)*y_pred[_][32+8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1C0*sum((-1)**a*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals)*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1C1*sum((-1)**a*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals)*sum((-1)**c*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B0C0*sum((-1)**(b+c)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B0C1*sum((-1)**(b+c)*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B1C0*sum((-1)**(b+c)*y_pred[_][8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B1C1*sum((-1)**(b+c)*y_pred[_][16+8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B0C0*sum((-1)**(a+b+c)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B0C1*sum((-1)**(a+b+c)*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B1C0*sum((-1)**(a+b+c)*y_pred[_][8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B1C1*sum((-1)**(a+b+c)*y_pred[_][16+8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B0C0*sum((-1)**(a+b+c)*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B0C1*sum((-1)**(a+b+c)*y_pred[_][32+16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B1C0*sum((-1)**(a+b+c)*y_pred[_][32+8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B1C1*sum((-1)**(a+b+c)*y_pred[_][32+16+8+a*4+b*2+c] for a in vals for b in vals for c in vals))
        loss_.append(-reduce_min(tf.linalg.eigvalsh(Gamma)))
    tf.print(K.sum(I), K.sum(loss_))
    #return K.sum(loss_)
    return tf.cond(tf.math.greater(K.sum(loss_), 0.), lambda: K.sum(loss_), lambda: -K.sum(I))

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

visible = Input(shape=(49,))
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
output9 = Dense(49, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
output=tf.concat([output1, output2,output3,output4,output5, output6,output7,output8,output9], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.00001, momentum=0.8)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)

input_=pd.read_table('coeff_bilocality_1.txt', delimiter=" ", header=None)
input_=input_.to_numpy()
input_=input_.astype('float32')
#input_=np.concatenate((input_, np.zeros((100,49))), axis=0)
#input_=np.zeros((20000,49))
train=tf.constant(input_)

callback=early_stop()
model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0)

model.save('bilocality')

'''
               A0*sum((-1)**a*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1*sum((-1)**a*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B0*sum((-1)**b*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B1*sum((-1)**b*y_pred[_][8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               C0*sum((-1)**c*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               C1*sum((-1)**c*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B0*sum((-1)**(a+b)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B1*sum((-1)**(a+b)*y_pred[_][8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0C0*sum((-1)**(a+c)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0C1*sum((-1)**(a+c)*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B0*sum((-1)**(a+b)*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B1*sum((-1)**(a+b)*y_pred[_][32+8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1C0*sum((-1)**(a+c)*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1C1*sum((-1)**(a+c)*y_pred[_][32+16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B0C0*sum((-1)**(b+c)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B0C1*sum((-1)**(b+c)*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B1C0*sum((-1)**(b+c)*y_pred[_][8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               B1C1*sum((-1)**(b+c)*y_pred[_][16+8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B0C0*sum((-1)**(a+b+c)*y_pred[_][a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B0C1*sum((-1)**(a+b+c)*y_pred[_][16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B1C0*sum((-1)**(a+b+c)*y_pred[_][8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A0B1C1*sum((-1)**(a+b+c)*y_pred[_][16+8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B0C0*sum((-1)**(a+b+c)*y_pred[_][32+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B0C1*sum((-1)**(a+b+c)*y_pred[_][32+16+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B1C0*sum((-1)**(a+b+c)*y_pred[_][32+8+a*4+b*2+c] for a in vals for b in vals for c in vals)+
               A1B1C1*sum((-1)**(a+b+c)*y_pred[_][32+16+8+a*4+b*2+c] for a in vals for b in vals for c in vals))
'''

'''
    for _ in range(1):
        I1=0.25*sum((-1)**(a+b+c)*(y_pred[_][x*8+a*2+b]*y_pred[_][16+y*2+b1*2+c]) for x in vals for y in vals for a in vals for b in vals for b1 in vals for c in vals)
        I2=0.25*sum((-1)**(a+b+c+x+y)*(y_pred[_][x*8+4+a*2+b]*y_pred[_][16+8+4*y+b1*2+c]) for x in vals for y in vals for a in vals for b in vals for b1 in vals for c in vals)
        I.append(K.sqrt(K.abs(I1))+K.sqrt(K.abs(I2)))
        Gamma=(Id+A0A1*(y_pred[_][32]+train[_][0])+B0B1*(y_pred[_][33]+train[_][2])+C0C1*(y_pred[_][66]+train[_][3])+
               A0A1C1C0*(y_pred[_][32]*y_pred[_][34]+train[_][1])+A0A1A0*(y_pred[_][67]+train[_][4])+A1A0A1*(y_pred[_][36]+train[_][5])+B0B1B0*(y_pred[_][37]+train[_][6])+
               B1B0B1*(y_pred[_][38]+train[_][7])+B0B1B0C0*(y_pred[_][39]+train[_][32])+C0C1C0*(y_pred[_][40]+train[_][8])+C1C0C1*(y_pred[_][41]+train[_][9])+A0B0B1*(y_pred[_][42]+train[_][10])+A1B0B1*(y_pred[_][43]+train[_][11])+
               A0C0C1*(y_pred[_][34]*sum((-1)**a*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][12])+
               A1C0C1*(y_pred[_][34]*sum((-1)**a*(y_pred[_][8+a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][13])+
               B1B0B1C0*(y_pred[_][44]+train[_][36])+B1B0B1C1*(y_pred[_][45]+train[_][37])+
               B0C0C1*(y_pred[_][46]+train[_][14])+B1C0C1*(y_pred[_][47]+train[_][15])+A0A1B0*(y_pred[_][48]+train[_][16])+A0A1B1*(y_pred[_][49]+train[_][17])+
               A0A1C0*(y_pred[_][32]*sum((-1)**c*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][18])+
               A0A1C1*(y_pred[_][32]*sum((-1)**c*(y_pred[_][4+a*2+b]*y_pred[_][16+8+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][19])+
               B0B1C1C0*(y_pred[_][49]+train[_][62])+B0B1C0C1*(y_pred[_][50]+train[_][59])+A0C0B0B1*(y_pred[_][51]+train[_][57])+
               B0B1C0*(y_pred[_][52]+train[_][20])+B0B1C1*(y_pred[_][53]+train[_][21])+A0A1A0B0*(y_pred[_][54]+train[_][22])+A0A1A0B1*(y_pred[_][55]+train[_][23])+
               A0A1A0C0*(y_pred[_][35]*sum((-1)**c*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][24])+
               A0A1A0C1*(y_pred[_][35]*sum((-1)**c*(y_pred[_][4+a*2+b]*y_pred[_][16+8+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][25])+
               A1A0A1B0*(y_pred[_][52]+train[_][26])+A1A0A1B1*(y_pred[_][53]+train[_][27])+
               A1A0A1C0*(y_pred[_][68]*sum((-1)**c*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][28])+
               A1A0A1C1*(y_pred[_][36]*sum((-1)**c*(y_pred[_][4+a*2+b]*y_pred[_][16+8+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][29])+
               B0B1B0A0*(y_pred[_][54]+train[_][30])+B0B1B0A1*(y_pred[_][55]+train[_][31])+B0B1B0C1*(y_pred[_][56]+train[_][33])+B1B0B1A0*(y_pred[_][57]+train[_][34])+
               B1B0B1A1*(y_pred[_][58]+train[_][35])+C0C1C0B0*(y_pred[_][59]+train[_][40])+C0C1C0B1*(y_pred[_][60]+train[_][41])+
               C0C1C0A0*(y_pred[_][40]*sum((-1)**a*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][38])+
               C0C1C0A1*(y_pred[_][40]*sum((-1)**a*(y_pred[_][8+a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][39])+
               C1C0C1A0*(y_pred[_][41]*sum((-1)**a*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][42])+
               C1C0C1A1*(y_pred[_][41]*sum((-1)**a*(y_pred[_][8+a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+train[_][43])+
               C1C0C1B0*(y_pred[_][61]+train[_][44])+C1C0C1B1*(y_pred[_][62]+train[_][45])+
               A0B0C0C1*(y_pred[_][63]+train[_][46])+A0B1C0C1*(y_pred[_][64]+train[_][47])+
               A1B0C0C1*(y_pred[_][65]+train[_][48])+A1B1C0C1*(y_pred[_][66]+train[_][49])+A0A1B0B1*(y_pred[_][67]+train[_][50])+
               A0A1C0C1*(y_pred[_][32]*y_pred[_][34]+train[_][51])+
               A0A1B1C0*(y_pred[_][68]+train[_][52])+A0A1B0C0*(y_pred[_][69]+train[_][53])+
               A0A1B0C1*(y_pred[_][70]+train[_][54])+A0A1B1B0*(y_pred[_][71]+train[_][55])+A0A1B1C1*(y_pred[_][72]+train[_][56])+
               A0C1B0B1*(y_pred[_][73]+train[_][58])+A1C1B0B1*(y_pred[_][74]+train[_][60])+A1C0B0B1*(y_pred[_][75]+train[_][61])+
               A0*sum((-1)**a*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A1*sum((-1)**a*(y_pred[_][8+a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               B0*sum((-1)**b*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               B1*sum((-1)**b*(y_pred[_][a*2+b]*y_pred[_][16+4+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               C0*sum((-1)**c*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               C1*sum((-1)**c*(y_pred[_][4+a*2+b]*y_pred[_][16+8+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A0B0*sum((-1)**(a+b)*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A0B1*sum((-1)**(a+b)*(y_pred[_][a*2+b]*y_pred[_][16+4+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A0C0*sum((-1)**(a+c)*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A0C1*sum((-1)**(a+c)*(y_pred[_][4+a*2+b]*y_pred[_][16+8+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A1B0*sum((-1)**(a+b)*(y_pred[_][8+a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A1B1*sum((-1)**(a+b)*(y_pred[_][8+a*2+b]*y_pred[_][16+4+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A1C0*sum((-1)**(a+c)*(y_pred[_][8+a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A1C1*sum((-1)**(a+c)*(y_pred[_][8+4+a*2+b]*y_pred[_][16+8+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               B0C0*sum((-1)**(b+c)*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               B0C1*sum((-1)**(b+c)*(y_pred[_][4+a*2+b]*y_pred[_][16+8+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               B1C0*sum((-1)**(b+c)*(y_pred[_][a*2+b]*y_pred[_][16+4+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               B1C1*sum((-1)**(b+c)*(y_pred[_][4+a*2+b]*y_pred[_][16+8+4+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A0B0C0*sum((-1)**(a+b+c)*(y_pred[_][a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A0B0C1*sum((-1)**(a+b+c)*(y_pred[_][4+a*2+b]*y_pred[_][16+8+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A0B1C0*sum((-1)**(a+b+c)*(y_pred[_][a*2+b]*y_pred[_][16+4+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A0B1C1*sum((-1)**(a+b+c)*(y_pred[_][4+a*2+b]*y_pred[_][16+8+4+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A1B0C0*sum((-1)**(a+b+c)*(y_pred[_][8+a*2+b]*y_pred[_][16+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A1B0C1*sum((-1)**(a+b+c)*(y_pred[_][8+4+a*2+b]*y_pred[_][16+8+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A1B1C0*sum((-1)**(a+b+c)*(y_pred[_][8+a*2+b]*y_pred[_][16+4+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals)+
               A1B1C1*sum((-1)**(a+b+c)*(y_pred[_][8+4+a*2+b]*y_pred[_][16+8+4+b1*2+c]) for a in vals for b in vals for b1 in vals for c in vals))
'''
