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
import sys

def custom_loss(train, y_pred):
    Id=np.eye(22)
    #elementi fisici
    A0B0C0=np.zeros((22,22))
    A0B1C0=np.zeros((22,22))
    A0B0C1=np.zeros((22,22))
    A0B1C1=np.zeros((22,22))
    A1B0C0=np.zeros((22,22))
    A1B1C0=np.zeros((22,22))
    A1B0C0=np.zeros((22,22))
    A1B0C1=np.zeros((22,22))
    A1B1C1=np.zeros((22,22))

    A0B0C0[13,1]=A0B0C0[1,13]=A0B0C0[10,3]=A0B0C0[3,10]=A0B0C0[8,5]=A0B0C0[5,8]=1.
    A0B1C0[15,1]=A0B1C0[1,15]=A0B1C0[10,4]=A0B1C0[4,10]=A0B1C0[9,5]=A0B1C0[5,9]=1.
    A0B0C1[14,1]=A0B0C1[1,14]=A0B0C1[11,3]=A0B0C1[3,11]=A0B0C1[8,6]=A0B0C1[6,8]=1.
    A0B1C1[16,1]=A0B1C1[1,16]=A0B1C1[11,4]=A0B1C1[4,11]=A0B1C1[9,6]=A0B1C1[6,9]=1.
    A1B0C0[13,2]=A1B0C0[2,13]=A1B0C0[20,3]=A1B0C0[3,20]=A1B0C0[18,5]=A1B0C0[5,18]=1.
    A1B1C0[15,2]=A1B1C0[2,15]=A1B1C0[20,4]=A1B1C0[4,20]=A1B1C0[19,5]=A1B1C0[5,19]=1.
    A1B0C1[14,2]=A1B0C1[2,14]=A1B0C1[21,3]=A1B0C1[3,21]=A1B0C1[18,6]=A1B0C1[6,18]=1.
    A1B1C1[16,2]=A1B1C1[2,16]=A1B1C1[21,4]=A1B1C1[4,21]=A1B1C1[19,6]=A1B1C1[6,19]=1.

    B0B1a=np.zeros((22,22))
    B0B1ab=np.zeros((22,22))
    B0B1abc=np.zeros((22,22))
    B0B1abcd=np.zeros((22,22))
    B0B1abcde=np.zeros((22,22))
    A0A1a=np.zeros((22,22))
    A0A1ab=np.zeros((22,22))
    A0A1abc=np.zeros((22,22))
    A0A1abcd=np.zeros((22,22))
    A0A1abcde=np.zeros((22,22))
    C0C1a=np.zeros((22,22))
    C0C1ab=np.zeros((22,22))
    C0C1abc=np.zeros((22,22))
    C0C1abcd=np.zeros((22,22))
    C0C1abcde=np.zeros((22,22))
    A0B0B1a=np.zeros((22,22))
    A0B0B1ab=np.zeros((22,22))
    A1B0B1a=np.zeros((22,22))
    A1B0B1ab=np.zeros((22,22))
    A0C0C1a=np.zeros((22,22))
    A0C0C1ab=np.zeros((22,22))
    A1C0C1a=np.zeros((22,22))
    A1C0C1ab=np.zeros((22,22))
    B0C0C1a=np.zeros((22,22))
    B0C0C1ab=np.zeros((22,22))
    B1C0C1a=np.zeros((22,22))
    B1C0C1ab=np.zeros((22,22))
    A0A1B0a=np.zeros((22,22))
    A0A1B0ab=np.zeros((22,22))
    A0A1B1a=np.zeros((22,22))
    A0A1B1ab=np.zeros((22,22))
    A0A1C0a=np.zeros((22,22))
    A0A1C0ab=np.zeros((22,22))
    A0A1C1a=np.zeros((22,22))
    A0A1C1ab=np.zeros((22,22))
    B0B1C0a=np.zeros((22,22))
    B0B1C0ab=np.zeros((22,22))
    B0B1C1a=np.zeros((22,22))
    B0B1C1ab=np.zeros((22,22))
    A0B0C0C1a = np.zeros((22,22))
    A0B0C0C1a = np.zeros((22,22))
    A0B0C0C1ab= np.zeros((22,22))
    A0B0C0C1ab= np.zeros((22,22))
    A0B1C0C1a= np.zeros((22,22))
    A0B1C0C1a= np.zeros((22,22))
    A0B1C0C1ab= np.zeros((22,22))
    A0B1C0C1ab= np.zeros((22,22))
    A1B0C0C1a= np.zeros((22,22))
    A1B0C0C1a= np.zeros((22,22))
    A1B0C0C1ab= np.zeros((22,22))
    A1B0C0C1ab= np.zeros((22,22))
    A1B1C0C1a= np.zeros((22,22))
    A1B1C0C1a= np.zeros((22,22))
    A1B1C0C1ab= np.zeros((22,22))
    A1B1C0C1ab= np.zeros((22,22))
    #A0A1B0B1a= np.zeros((22,22))
    #A0A1B0B1a= np.zeros((22,22))
    A0A1C0C1a= np.zeros((22,22))
    A0A1C0C1a= np.zeros((22,22))
    B0B1C0C1a= np.zeros((22,22))
    B0B1C0C1a= np.zeros((22,22))
    A0A1B1C0a= np.zeros((22,22))
    A0A1B1C0ab= np.zeros((22,22))
    A0A1B1C1a= np.zeros((22,22))
    A0A1B1C1ab= np.zeros((22,22))
    A0A1B0C0a= np.zeros((22,22))
    A0A1B0C0ab= np.zeros((22,22))
    A0A1B0C1a= np.zeros((22,22))
    A0A1B0C1ab= np.zeros((22,22))
    A0C0B0B1a= np.zeros((22,22))
    A0C0B0B1ab= np.zeros((22,22))
    A0C1B0B1a= np.zeros((22,22))
    A0C1B0B1ab= np.zeros((22,22))
    A1C1B0B1a= np.zeros((22,22))
    A1C1B0B1ab= np.zeros((22,22))
    A1C0B0B1a= np.zeros((22,22))
    A1C0B0B1ab= np.zeros((22,22))
    B0C1a=np.zeros((22,22))
    B0C1ab=np.zeros((22,22))
    B0C1abc=np.zeros((22,22))
    B1C0a=np.zeros((22,22))
    B1C0ab=np.zeros((22,22))
    B1C0abc=np.zeros((22,22))
    B1C1a=np.zeros((22,22))
    B1C1ab=np.zeros((22,22))
    B1C1abc=np.zeros((22,22))
    A0B0C0a=np.zeros((22,22))
    A0B0C0ab=np.zeros((22,22))
    A0B1C0a=np.zeros((22,22))
    A0B1C0ab=np.zeros((22,22))
    A0B0C1a=np.zeros((22,22))
    A0B0C1ab=np.zeros((22,22))
    A0B1C1a=np.zeros((22,22))
    A0B1C1ab=np.zeros((22,22))
    A1B0C0a=np.zeros((22,22))
    A1B0C0ab=np.zeros((22,22))
    A1B0C1a=np.zeros((22,22))
    A1B0C1ab=np.zeros((22,22))
    A1B1C0a=np.zeros((22,22))
    A1B1C0ab=np.zeros((22,22))
    A1B1C1a=np.zeros((22,22))
    A1B1C1ab=np.zeros((22,22))
    A0a= np.zeros([22,22])
    A0ab= np.zeros([22,22])
    A0abc= np.zeros([22,22])
    A0abcd= np.zeros([22,22])
    A1a= np.zeros([22,22])
    A1ab= np.zeros([22,22])
    A1abc= np.zeros([22,22])
    A1abcd= np.zeros([22,22])
    B0a= np.zeros([22,22])
    B0ab= np.zeros([22,22])
    B0abc= np.zeros([22,22])
    B0abcd= np.zeros([22,22])
    B1a= np.zeros([22,22])
    B1ab= np.zeros([22,22])
    B1abc= np.zeros([22,22])
    B1abcd= np.zeros([22,22])
    C0a= np.zeros([22,22])
    C0ab= np.zeros([22,22])
    C0abc= np.zeros([22,22])
    C0abcd= np.zeros([22,22])
    C1a= np.zeros([22,22])
    C1ab= np.zeros([22,22])
    C1abc= np.zeros([22,22])
    C1abcd= np.zeros([22,22])
    A0B0a= np.zeros([22,22])
    A0B0ab= np.zeros([22,22])
    A0B0abc= np.zeros([22,22])
    A0B1a= np.zeros([22,22])
    A0B1ab= np.zeros([22,22])
    A0B1abc= np.zeros([22,22])
    A0C0a= np.zeros([22,22])
    A0C0ab= np.zeros([22,22])
    A0C0abc= np.zeros([22,22])
    A0C1a= np.zeros([22,22])
    A0C1ab= np.zeros([22,22])
    A0C1abc= np.zeros([22,22])
    A1B0a= np.zeros([22,22])
    A1B0ab= np.zeros([22,22])
    A1B0abc= np.zeros([22,22])
    A1B1a= np.zeros([22,22])
    A1B1ab= np.zeros([22,22])
    A1B1abc= np.zeros([22,22])
    A1C0a= np.zeros([22,22])
    A1C0ab= np.zeros([22,22])
    A1C0abc= np.zeros([22,22])
    A1C1a= np.zeros([22,22])
    A1C1ab= np.zeros([22,22])
    A1C1abc= np.zeros([22,22])
    B0C0a= np.zeros([22,22])
    B0C0ab= np.zeros([22,22])
    B0C0abc= np.zeros([22,22])

    B0B1a[4,3]=B0B1a[3,4]=1.
    B0B1a[9,8]=B0B1a[8,9]=-1.
    B0B1ab[4,3]=B0B1ab[3,4]=1.
    B0B1ab[12,0]=B0B1ab[0,12]=-1.
    B0B1abc[4,3]=B0B1abc[3,4]=1.
    B0B1abc[15,13]=B0B1abc[13,15]=-1.
    B0B1abcd[4,3]=B0B1abcd[3,4]=1.
    B0B1abcd[16,14]=B0B1abcd[14,16]=-1.
    B0B1abcde[4,3]=B0B1abcde[3,4]=1.
    B0B1abcde[19,18]=B0B1abcde[18,19]=-1.
    A0A1a[8,18] = A0A1a[18,8] = 1.
    A0A1a[0,7]  = A0A1a[7,0] = -1.
    A0A1ab[19,9] = A0A1ab[9,19] = 1.
    A0A1ab[0,7]  = A0A1ab[7,0] = -1.
    A0A1abc[20,10] = A0A1abc[10,20] = 1.
    A0A1abc[0,7] =   A0A1abc[7,0] = -1.
    A0A1abcd[21,11] = A0A1abc[11,21] = 1
    A0A1abcd[0,7] =   A0A1abc[7,0] = -1.
    A0A1abcde[2,1] = A0A1abcde[1,2] = 1.
    A0A1abcde[7,0] = A0A1abcde[0,7] = -1.
    C0C1a[5,6] = C0C1a[6,5] = 1.
    C0C1a[0,17]  = C0C1a[17,0] = -1.
    C0C1ab[11,10] = C0C1ab[10,11] = 1.
    C0C1ab[0,17]  = C0C1ab[17,0] = -1.
    C0C1abc[14,13] = C0C1abc[13,14] = 1.
    C0C1abc[0,17] =   C0C1abc[17,0] = -1.
    C0C1abcd[16,15] = C0C1abc[15,16] = 1
    C0C1abcd[0,17] =   C0C1abc[17,0] = -1.
    C0C1abcde[21,20] = C0C1abcde[20,21] = 1.
    C0C1abcde[17,0] = C0C1abcde[0,17] = -1.    
    A0B0B1a[12,1]=A0B0B1a[1,12]=1.
    A0B0B1a[9,3]=A0B0B1a[3,9]=-1.
    A0B0B1ab[12,1]=A0B0B1ab[1,12]=1.
    A0B0B1ab[8,4]=A0B0B1ab[4,8]=-1.
    A1B0B1a[12,2]=A1B0B1a[2,12]=1.    
    A1B0B1a[19,3]=A1B0B1a[3,19]=-1.    
    A1B0B1ab[12,2]=A1B0B1ab[2,12]=1.    
    A1B0B1ab[18,4]=A1B0B1ab[4,18]=-1.    
    A0C0C1a[11,5]=A0C0C1a[5,11]=1.
    A0C0C1a[17,1]=A0C0C1a[1,17]=-1.
    A0C0C1ab[11,5]=A0C0C1ab[5,11]=1.
    A0C0C1ab[10,6]=A0C0C1ab[6,10]=-1.
    A1C0C1a[21,5]=A1C0C1a[5,21]=1.
    A1C0C1a[17,2]=A1C0C1a[2,17]=-1.
    A1C0C1ab[21,5]=A1C0C1ab[5,21]=1.
    A1C0C1ab[20,6]=A1C0C1ab[6,20]=-1.
    B0C0C1a[13,6]=B0C0C1a[6,13]=1.
    B0C0C1a[14,5]=B0C0C1a[5,14]=-1.
    B0C0C1ab[13,6]=B0C0C1ab[6,13]=1.
    B0C0C1ab[17,3]=B0C0C1ab[3,17]=-1.
    B1C0C1a[15,6]=B1C0C1a[6,15]=1.
    B1C0C1a[16,5]=B1C0C1a[5,16]=-1.
    B1C0C1ab[15,6]=B1C0C1ab[6,15]=1.
    B1C0C1ab[17,4]=B1C0C1ab[4,17]=-1.
    A0A1B0a[8,2]=A0A1B0a[2,8]=1.
    A0A1B0a[7,3]=A0A1B0a[3,7]=-1.
    A0A1B0ab[8,2]=A0A1B0ab[2,8]=1.
    A0A1B0ab[18,1]=A0A1B0ab[1,18]=-1.
    A0A1B1a[9,2]=A0A1B1a[2,9]=1.
    A0A1B1a[7,4]=A0A1B1a[4,7]=-1.
    A0A1B1ab[9,2]=A0A1B1ab[2,9]=1.
    A0A1B1ab[19,1]=A0A1B1ab[1,19]=-1.
    A0A1C0a[10,2]=A0A1C0a[2,10]=1.
    A0A1C0a[7,5]=A0A1C0a[5,7]=-1.
    A0A1C0ab[10,2]=A0A1C0ab[2,10]=1.
    A0A1C0ab[20,1]=A0A1C0ab[1,20]=-1.
    A0A1C1a[11,2]=A0A1C1a[2,11]=1.
    A0A1C1a[7,6]=A0A1C1a[6,7]=-1.
    A0A1C1ab[11,2]=A0A1C1ab[2,11]=1.
    A0A1C1ab[21,1]=A0A1C1ab[1,21]=-1.
    B0B1C0a[13,4]=B0B1C0a[4,13]=1.
    B0B1C0a[12,5]=B0B1C0a[5,12]=-1.
    B0B1C0ab[13,4]=B0B1C0ab[4,13]=1.
    B0B1C0ab[15,3]=B0B1C0ab[3,15]=-1.
    B0B1C1a[14,4]=B0B1C1a[4,14]=1.
    B0B1C1a[12,6]=B0B1C1a[6,12]=-1.
    B0B1C1ab[14,4]=B0B1C1ab[4,14]=1.
    B0B1C1ab[16,3]=B0B1C1ab[3,16]=-1.    
    A0B0C0C1a[13,11] = A0B0C0C1a[11,13] = -1.
    A0B0C0C1a[14,10] = A0B0C0C1a[10,14] =  1.
    A0B0C0C1ab[17,8] =  A0B0C0C1ab[8,17]  = -1.
    A0B0C0C1ab[14,10] = A0B0C0C1ab[10,14] =  1.
    A0B1C0C1a[15,11]=A0B1C0C1a[11,15] = -1.
    A0B1C0C1a[16,10]=A0B1C0C1a[10,16] = 1.
    A0B1C0C1ab[15,11]= A0B1C0C1ab[11,15] = -1.
    A0B1C0C1ab[17,9] = A0B1C0C1ab[9,17] = 1.    
    A1B0C0C1a[18,17]=A1B0C0C1a[17,18] = 1.
    A1B0C0C1a[20,14]=A1B0C0C1a[14,20] = -1.
    A1B0C0C1ab[18,17]=A1B0C0C1ab[17,18] = 1.
    A1B0C0C1ab[21,13]=A1B0C0C1ab[13,21] = -1.
    A1B1C0C1a[19,17]=A1B1C0C1a[17,19] = 1.
    A1B1C0C1a[20,16]=A1B1C0C1a[16,20] = -1.
    A1B1C0C1ab[19,17]=A1B1C0C1ab[17,19] = 1.
    A1B1C0C1ab[21,15]=A1B1C0C1ab[15,21] = -1.
    #A0A1B0B1a[12,7]=A0A1B0B1a[7,12] = 1.
    #A0A1B0B1a[19,8]=A0A1B0B1a[8,19] = -1
    A0A1C0C1a[17,7]= A0A1C0C1a[7,17] = 1.
    A0A1C0C1a[21,10]=A0A1C0C1a[10,21] = -1.
    B0B1C0C1a[16,13]=B0B1C0C1a[13,16] = 1.
    B0B1C0C1a[17,12]=B0B1C0C1a[12,17] = -1.
    A0A1B1C0a[19,10]=A0A1B1C0a[10,19] = 1.
    A0A1B1C0a[15,7]= A0A1B1C0a[7,15] = -1.
    A0A1B1C0ab[19,10]=A0A1B1C0ab[10,19] = 1.
    A0A1B1C0ab[20,9]=A0A1B1C0ab[9,20] = -1.
    A0A1B1C1a[19,11]=A0A1B1C1a[11,19] = 1.
    A0A1B1C1a[21,9] =A0A1B1C1a[9,21] = -1.
    A0A1B1C1ab[19,11]=A0A1B1C1ab[11,19] = 1.
    A0A1B1C1ab[16,7]= A0A1B1C1ab[7,16] = -1.    
    A0A1B0C0a[13,7]= A0A1B0C0a[7,13] = 1.
    A0A1B0C0a[18,10]=A0A1B0C0a[10,18] = -1.
    A0A1B0C0ab[13,7]=A0A1B0C0ab[7,13] = 1.
    A0A1B0C0ab[20,8]=A0A1B0C0ab[8,20] = -1.
    A0A1B0C1a[18,11]=A0A1B0C1a[11,18] = 1.
    A0A1B0C1a[21,8]= A0A1B0C1a[8,21] = -1.
    A0A1B0C1ab[18,11]=A0A1B0C1ab[11,18] = 1.
    A0A1B0C1ab[14,7]= A0A1B0C1ab[7,14] = -1.
    A0C0B0B1a[12,10]=A0C0B0B1a[10,12] = 1.
    A0C0B0B1a[13,9]= A0C0B0B1a[9,13] = -1.
    A0C0B0B1ab[12,10]=A0C0B0B1ab[10,12] = 1.
    A0C0B0B1ab[15,8]= A0C0B0B1ab[8,15] = -1.
    A0C1B0B1a[12,11]=A0C1B0B1a[11,12] = 1.
    A0C1B0B1a[14,9]= A0C1B0B1a[9,14] = -1.
    A0C1B0B1ab[12,11]=A0C1B0B1ab[11,12] = 1.
    A0C1B0B1ab[16,8]= A0C1B0B1ab[8,16] = -1.
    A1C1B0B1a[18,16]=A1C1B0B1a[16,18] = 1.
    A1C1B0B1a[19,14]=A1C1B0B1a[14,19] = -1.
    A1C1B0B1ab[18,16]=A1C1B0B1ab[16,18] = 1.
    A1C1B0B1ab[21,12]=A1C1B0B1ab[12,21] = -1.
    A1C0B0B1a[18,15]=A1C0B0B1a[15,18] = 1.
    A1C0B0B1a[19,13]=A1C0B0B1a[13,19] = -1.
    A1C0B0B1ab[18,15]=A1C0B0B1ab[15,18] = 1.
    A1C0B0B1ab[20,12]=A1C0B0B1ab[12,20] = -1.    
    B0C1a[14,0]=B0C1a[0,14]=1.
    B0C1a[6,3]=B0C1a[3,6]=-1.
    B0C1ab[0,14]=B0C1ab[14,0]=1.
    B0C1ab[11,8]=B0C1ab[8,11]=-1.
    B0C1abc[14,0]=B0C1abc[0,14]=1.
    B0C1abc[21,18]=B0C1abc[18,21]=1.
    B1C0a[15,0]=B1C0a[0,15]=1.
    B1C0a[5,4]=B1C0a[4,5]=-1.
    B1C0ab[0,15]=B1C0ab[15,0]=1.
    B1C0ab[10,9]=B1C0ab[9,10]=-1.
    B1C0abc[16,0]=B1C0abc[0,16]=1.
    B1C0abc[19,21]=B1C0abc[21,19]=-1.
    B1C1a[16,0]=B1C1a[0,16]=1.
    B1C1a[6,4]=B1C1a[4,6]=-1.
    B1C1ab[0,16]=B1C1ab[16,0]=1.
    B1C1ab[11,9]=B1C1ab[9,11]=-1.
    B1C1abc[16,0]=B1C1abc[0,16]=1.
    B1C1abc[19,21]=B1C1abc[21,19]=-1.
    A0B0C0a[13,1]=A0B0C0a[1,13]=1.
    A0B0C0a[10,3]=A0B0C0a[3,10]=-1.
    A0B0C0ab[13,1]=A0B0C0ab[1,13]=1.
    A0B0C0ab[8,5]=A0B0C0ab[5,8]=-1.    
    A0B1C0a[15,1]=A0B1C0a[1,15]=1.
    A0B1C0a[10,4]=A0B1C0a[4,10]=-1.
    A0B1C0ab[15,1]=A0B1C0ab[1,15]=1.
    A0B1C0ab[9,5]=A0B1C0ab[5,9]=-1.    
    A0B0C1a[14,1]=A0B0C1a[1,14]=1.
    A0B0C1a[11,3]=A0B0C1a[3,11]=-1.
    A0B0C1ab[14,1]=A0B0C1ab[1,14]=1.
    A0B0C1ab[8,6]=A0B0C1ab[6,8]=-1.    
    A0B1C1a[16,1]=A0B1C1a[1,16]=1.
    A0B1C1a[11,4]=A0B1C1a[4,11]=-1.
    A0B1C1ab[16,1]=A0B1C1ab[1,16]=1.
    A0B1C1ab[9,6]=A0B1C1ab[6,9]=-1.    
    A1B0C0a[13,2]=A1B0C0a[2,13]=1.
    A1B0C0a[20,3]=A1B0C0a[3,20]=-1.
    A1B0C0ab[13,2]=A1B0C0ab[2,13]=1.
    A1B0C0ab[18,5]=A1B0C0ab[5,18]=-1.
    A1B1C0a[15,2]=A1B1C0a[2,15]=1.
    A1B1C0a[20,4]=A1B1C0a[4,20]=-1.
    A1B1C0ab[15,2]=A1B1C0ab[2,15]=1.
    A1B1C0ab[19,5]=A1B1C0ab[5,19]=-1.
    A1B0C1a[14,2]=A1B0C1a[2,14]=1.
    A1B0C1a[21,3]=A1B0C1a[3,21]=-1.
    A1B0C1ab[14,2]=A1B0C1ab[2,14]=1.
    A1B0C1ab[18,6]=A1B0C1ab[6,18]=-1.
    A1B1C1a[16,2]=A1B1C1a[2,16]=1.
    A1B1C1a[21,4]=A1B1C1a[4,21]=-1.
    A1B1C1ab[16,2]=A1B1C1ab[2,16]=1.
    A1B1C1ab[19,6]=A1B1C1ab[6,19]=-1.
    vals=[0,1]
    loss_=[]
    C=A0B0C0/np.sum(A0B0C0)+A0B0C1/np.sum(A0B0C1)+A0B1C0/np.sum(A0B1C0)-A0B1C1/np.sum(A0B1C1)+A1B0C0/np.sum(A1B0C0)-A1B1C0/np.sum(A1B1C0)-A1B0C1/np.sum(A1B0C1)-A1B1C1/np.sum(A1B1C1)
    I=[]
    for _ in range(1):
        I.append((y_pred[_][154]+train[_][154])+
                 (y_pred[_][155]+train[_][155])+
                 (y_pred[_][156]+train[_][156])+
                 (y_pred[_][157]+train[_][157])+
                 (y_pred[_][158]+train[_][158])+
                 (y_pred[_][159]+train[_][159])+
                 (y_pred[_][160]+train[_][160])+
                 (y_pred[_][161]+train[_][161])+
                 (y_pred[_][162]+train[_][162])+
                 (y_pred[_][163]+train[_][163])+
                 (y_pred[_][164]+train[_][164])+
                 (y_pred[_][165]+train[_][165])+
                 (y_pred[_][166]+train[_][166])+
                 (y_pred[_][167]+train[_][167])+
                 (y_pred[_][168]+train[_][168])+
                 (y_pred[_][169]+train[_][169])+
                 (y_pred[_][170]+train[_][170])+
                 (y_pred[_][171]+train[_][171])+
                 (y_pred[_][172]+train[_][172])+
                 (y_pred[_][173]+train[_][173])+
                 (y_pred[_][174]+train[_][174])+
                 (y_pred[_][175]+train[_][175]))
        Gamma=C-(np.diag([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][154]+train[_][154])+
                    np.diag([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][155]+train[_][155])+
                    np.diag([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][156]+train[_][156])+
                    np.diag([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][157]+train[_][157])+
                    np.diag([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][158]+train[_][158])+
                    np.diag([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][159]+train[_][159])+
                    np.diag([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][160]+train[_][160])+
                    np.diag([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][161]+train[_][161])+
                    np.diag([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][162]+train[_][162])+
                    np.diag([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][163]+train[_][163])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][164]+train[_][164])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])*(y_pred[_][165]+train[_][165])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])*(y_pred[_][166]+train[_][166])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])*(y_pred[_][167]+train[_][167])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])*(y_pred[_][168]+train[_][168])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])*(y_pred[_][169]+train[_][169])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])*(y_pred[_][170]+train[_][170])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])*(y_pred[_][171]+train[_][171])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])*(y_pred[_][172]+train[_][172])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])*(y_pred[_][173]+train[_][173])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])*(y_pred[_][174]+train[_][174])+
                    np.diag([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])*(y_pred[_][175]+train[_][175])+
                    (y_pred[_][0]+train[_][0])*B0B1a+
                    (y_pred[_][1]+train[_][1])*B0B1ab+
                    (y_pred[_][2]+train[_][2])*B0B1abc+
                    (y_pred[_][3]+train[_][3])*B0B1abcd+
                    (y_pred[_][4]+train[_][4])*B0B1abcde+
                    (y_pred[_][5]+train[_][5])*A0A1a+
                    (y_pred[_][6]+train[_][6])*A0A1ab+
                    (y_pred[_][7]+train[_][7])*A0A1abc+
                    (y_pred[_][8]+train[_][8])*A0A1abcd+
                    (y_pred[_][9]+train[_][9])*A0A1abcde+
                    (y_pred[_][10]+train[_][10])*C0C1a+
                    (y_pred[_][11]+train[_][11])*C0C1ab+
                    (y_pred[_][12]+train[_][12])*C0C1abc+
                    (y_pred[_][13]+train[_][13])*C0C1abcd+
                    (y_pred[_][14]+train[_][14])*C0C1abcde+
                    (y_pred[_][15]+train[_][15])*A0B0B1a+
                    (y_pred[_][16]+train[_][16])*A0B0B1ab+
                    (y_pred[_][17]+train[_][17])*A1B0B1a+
                    (y_pred[_][18]+train[_][18])*A1B0B1ab+
                    (y_pred[_][19]+train[_][19])*A0C0C1a+
                    (y_pred[_][20]+train[_][20])*A0C0C1ab+
                    (y_pred[_][21]+train[_][21])*A1C0C1a+
                    (y_pred[_][22]+train[_][22])*A1C0C1ab+
                    (y_pred[_][23]+train[_][23])*B0C0C1a+
                    (y_pred[_][24]+train[_][24])*B0C0C1ab+
                    (y_pred[_][25]+train[_][25])*B1C0C1a+
                    (y_pred[_][26]+train[_][26])*B1C0C1ab+
                    (y_pred[_][27]+train[_][27])*A0A1B0a+
                    (y_pred[_][28]+train[_][28])*A0A1B0ab+
                    (y_pred[_][29]+train[_][29])*A0A1B1a+
                    (y_pred[_][30]+train[_][30])*A0A1B1ab+
                    (y_pred[_][31]+train[_][31])*A0A1C0a+
                    (y_pred[_][32]+train[_][32])*A0A1C0ab+
                    (y_pred[_][33]+train[_][33])*A0A1C1a+
                    (y_pred[_][34]+train[_][34])*A0A1C1ab+
                    (y_pred[_][35]+train[_][35])*B0B1C0a+
                    (y_pred[_][36]+train[_][36])*B0B1C0ab+
                    (y_pred[_][37]+train[_][37])*B0B1C1a+
                    (y_pred[_][38]+train[_][38])*B0B1C1ab+
                    (y_pred[_][39]+train[_][39])*A0B0C0C1a+
                    (y_pred[_][40]+train[_][40])*A0B0C0C1a+
                    (y_pred[_][41]+train[_][41])*A0B0C0C1ab+
                    (y_pred[_][42]+train[_][42])*A0B0C0C1ab+
                    (y_pred[_][43]+train[_][43])*A0B1C0C1a+
                    (y_pred[_][44]+train[_][44])*A0B1C0C1a+
                    (y_pred[_][45]+train[_][45])*A0B1C0C1ab+
                    (y_pred[_][46]+train[_][46])*A0B1C0C1ab+
                    (y_pred[_][47]+train[_][47])*A1B0C0C1a+
                    (y_pred[_][48]+train[_][48])*A1B0C0C1a+
                    (y_pred[_][49]+train[_][49])*A1B0C0C1ab+
                    (y_pred[_][50]+train[_][50])*A1B0C0C1ab+
                    (y_pred[_][51]+train[_][51])*A1B1C0C1a+
                    (y_pred[_][52]+train[_][52])*A1B1C0C1a+
                    (y_pred[_][53]+train[_][53])*A1B1C0C1ab+
                    (y_pred[_][54]+train[_][54])*A1B1C0C1ab+
                    #(y_pred[_][55]+train[_][55])*A0A1B0B1a+
                    #(y_pred[_][56]+train[_][56])*A0A1B0B1a+
                    (y_pred[_][57]+train[_][57])*A0A1C0C1a+
                    (y_pred[_][58]+train[_][58])*A0A1C0C1a+
                    (y_pred[_][59]+train[_][59])*B0B1C0C1a+
                    (y_pred[_][60]+train[_][60])*B0B1C0C1a+
                    (y_pred[_][61]+train[_][61])*A0A1B1C0a+
                    (y_pred[_][62]+train[_][62])*A0A1B1C0ab+
                    (y_pred[_][63]+train[_][63])*A0A1B1C1a+
                    (y_pred[_][64]+train[_][64])*A0A1B1C1ab+
                    (y_pred[_][65]+train[_][65])*A0A1B0C0a+
                    (y_pred[_][66]+train[_][66])*A0A1B0C0ab+
                    (y_pred[_][67]+train[_][67])*A0A1B0C1a+
                    (y_pred[_][68]+train[_][68])*A0A1B0C1ab+
                    (y_pred[_][69]+train[_][69])*A0C0B0B1a+
                    (y_pred[_][70]+train[_][70])*A0C0B0B1ab+
                    (y_pred[_][71]+train[_][71])*A0C1B0B1a+
                    (y_pred[_][72]+train[_][72])*A0C1B0B1ab+
                    (y_pred[_][73]+train[_][73])*A1C1B0B1a+
                    (y_pred[_][74]+train[_][74])*A1C1B0B1ab+
                    (y_pred[_][75]+train[_][75])*A1C0B0B1a+
                    (y_pred[_][76]+train[_][76])*A1C0B0B1ab+
                    (y_pred[_][77]+train[_][77])*B0C1a+
                    (y_pred[_][78]+train[_][78])*B0C1ab+
                    (y_pred[_][79]+train[_][79])*B0C1abc+
                    (y_pred[_][80]+train[_][80])*B1C0a+
                    (y_pred[_][81]+train[_][81])*B1C0ab+
                    (y_pred[_][82]+train[_][82])*B1C0abc+
                    (y_pred[_][83]+train[_][83])*B1C1a+
                    (y_pred[_][84]+train[_][84])*B1C1ab+
                    (y_pred[_][85]+train[_][85])*B1C1abc+
                    (y_pred[_][86]+train[_][86])*A0B0C0a+
                    (y_pred[_][87]+train[_][87])*A0B0C0ab+
                    (y_pred[_][88]+train[_][88])*A0B1C0a+
                    (y_pred[_][89]+train[_][89])*A0B1C0ab+
                    (y_pred[_][90]+train[_][90])*A0B0C1a+
                    (y_pred[_][91]+train[_][91])*A0B0C1ab+
                    (y_pred[_][92]+train[_][92])*A0B1C1a+
                    (y_pred[_][93]+train[_][93])*A0B1C1ab+
                    (y_pred[_][94]+train[_][94])*A1B0C0a+
                    (y_pred[_][95]+train[_][95])*A1B0C0ab+
                    (y_pred[_][96]+train[_][96])*A1B0C1a+
                    (y_pred[_][97]+train[_][97])*A1B0C1ab+
                    (y_pred[_][98]+train[_][98])*A1B1C0a+
                    (y_pred[_][99]+train[_][99])*A1B1C0ab+
                    (y_pred[_][100]+train[_][100])*A1B1C1a+
                    (y_pred[_][101]+train[_][101])*A1B1C1ab+
                    (y_pred[_][102]+train[_][102])*A0a+
                    (y_pred[_][103]+train[_][103])*A0ab+
                    (y_pred[_][104]+train[_][104])*A0abc+
                    (y_pred[_][105]+train[_][105])*A0abcd+
                    (y_pred[_][106]+train[_][106])*A1a+
                    (y_pred[_][107]+train[_][107])*A1ab+
                    (y_pred[_][108]+train[_][108])*A1abc+
                    (y_pred[_][109]+train[_][109])*A1abcd+
                    (y_pred[_][110]+train[_][110])*B0a+
                    (y_pred[_][111]+train[_][111])*B0ab+
                    (y_pred[_][112]+train[_][112])*B0abc+
                    (y_pred[_][113]+train[_][113])*B0abcd+
                    (y_pred[_][114]+train[_][114])*B1a+
                    (y_pred[_][115]+train[_][115])*B1ab+
                    (y_pred[_][116]+train[_][116])*B1abc+
                    (y_pred[_][117]+train[_][117])*B1abcd+
                    (y_pred[_][118]+train[_][118])*C0a+
                    (y_pred[_][119]+train[_][119])*C0ab+
                    (y_pred[_][120]+train[_][120])*C0abc+
                    (y_pred[_][121]+train[_][121])*C0abcd+
                    (y_pred[_][122]+train[_][122])*C1a+
                    (y_pred[_][123]+train[_][123])*C1ab+
                    (y_pred[_][124]+train[_][124])*C1abc+
                    (y_pred[_][125]+train[_][125])*C1abcd+
                    (y_pred[_][126]+train[_][126])*A0B0a+
                    (y_pred[_][127]+train[_][127])*A0B0ab+
                    (y_pred[_][128]+train[_][128])*A0B0abc+
                    (y_pred[_][129]+train[_][129])*A0B1a+
                    (y_pred[_][130]+train[_][130])*A0B1ab+
                    (y_pred[_][131]+train[_][131])*A0B1abc+
                    (y_pred[_][132]+train[_][132])*A0C0a+
                    (y_pred[_][133]+train[_][133])*A0C0ab+
                    (y_pred[_][134]+train[_][134])*A0C0abc+
                    (y_pred[_][135]+train[_][135])*A0C1a+
                    (y_pred[_][136]+train[_][136])*A0C1ab+
                    (y_pred[_][137]+train[_][137])*A0C1abc+
                    (y_pred[_][138]+train[_][138])*A1B0a+
                    (y_pred[_][139]+train[_][139])*A1B0ab+
                    (y_pred[_][140]+train[_][140])*A1B0abc+
                    (y_pred[_][141]+train[_][141])*A1B1a+
                    (y_pred[_][142]+train[_][142])*A1B1ab+
                    (y_pred[_][143]+train[_][143])*A1B1abc+
                    (y_pred[_][144]+train[_][144])*A1C0a+
                    (y_pred[_][145]+train[_][145])*A1C0ab+
                    (y_pred[_][146]+train[_][146])*A1C0abc+
                    (y_pred[_][147]+train[_][147])*A1C1a+
                    (y_pred[_][148]+train[_][148])*A1C1ab+
                    (y_pred[_][149]+train[_][149])*A1C1abc+
                    (y_pred[_][150]+train[_][150])*B0C0a+
                    (y_pred[_][151]+train[_][151])*B0C0ab+
                    (y_pred[_][152]+train[_][152])*B0C0abc)
        loss_.append(-reduce_min(tf.linalg.eigvalsh(Gamma)))
    tf.print(K.sum(I), K.sum(loss_))
    #return K.sum(loss_)
    return tf.cond(tf.math.greater(K.sum(loss_), 0.), lambda: K.sum(loss_), lambda: K.sum(loss_)-K.sum(I))

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

visible = Input(shape=(176,))
hidden1 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(visible)
hidden2 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden1)
hidden3 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden2)
hidden4 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden3)
hidden5 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden4)
hidden6 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden5)
hidden7 = Dense(1000, activation = 'elu', activity_regularizer=regularizers.l2(1e-6))(hidden6)
output = Dense(176, activation = 'tanh', activity_regularizer=regularizers.l2(1e-6))(hidden7)
#output=tf.concat([output1, output2,output3,output4,output5, output6,output7,output8,output9], axis=-1)
opt=keras.optimizers.SGD(learning_rate=0.00001, momentum=0.8)
#opt=keras.optimizers.Adamax(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model=Model(inputs = visible, outputs = output)
model.compile(optimizer=opt, loss=custom_loss)

#input_=pd.read_table('coeff_tripartite.txt', delimiter=" ", header=None)
#input_=input_.to_numpy()
#input_=input_.astype('float32')
#input_=np.concatenate((input_, np.zeros((100,71))), axis=0)
input_=np.zeros((40000,176))
input_=input_.astype('float32')
train=tf.constant(input_)

callback=early_stop()
model.fit(train, train, batch_size=1, epochs=1, shuffle=True, verbose=0)

model.save('dual_tripartite_4')



'''
        I.append(((y_pred[_][127]+train[_][63])*sum((-1)**a*p[a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A0)+
               (y_pred[_][128]+train[_][64])*sum((-1)**a*p[32+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A1)+
               (y_pred[_][129]+train[_][65])*sum((-1)**b*p[a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(B0)+
               (y_pred[_][130]+train[_][66])*sum((-1)**b*p[16+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(B1)+
               (y_pred[_][131]+train[_][67])*sum((-1)**c*p[a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(C0)+
               (y_pred[_][132]+train[_][68])*sum((-1)**c*p[8+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(C1)+
               (y_pred[_][133]+train[_][69])*sum((-1)**(a+b)*p[a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A0B0)+
               (y_pred[_][134]+train[_][70])*sum((-1)**(a+b)*p[16+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A0B1)+
               (y_pred[_][135]+train[_][71])*sum((-1)**(a+c)*p[a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A0C0)+
               (y_pred[_][136]+train[_][72])*sum((-1)**(a+c)*p[8+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A0C1)+
               (y_pred[_][137]+train[_][73])*sum((-1)**(a+b)*p[32+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A1B0)+
               (y_pred[_][138]+train[_][74])*sum((-1)**(a+b)*p[32+16+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A1B1)+
               (y_pred[_][139]+train[_][75])*sum((-1)**(a+c)*p[32+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A1C0)+
               (y_pred[_][140]+train[_][76])*sum((-1)**(a+c)*p[32+8+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A1C1)+
               (y_pred[_][141]+train[_][77])*sum((-1)**(b+c)*p[a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(B0C0)+
               (y_pred[_][142]+train[_][78])*sum((-1)**(b+c)*p[8+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(B0C1)+
               (y_pred[_][143]+train[_][79])*sum((-1)**(b+c)*p[16+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(B1C0)+
               (y_pred[_][144]+train[_][80])*sum((-1)**(b+c)*p[16+8+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(B1C1)+
               (y_pred[_][145]+train[_][81])*sum((-1)**(a+b+c)*p[a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A0B0C0)+
               (y_pred[_][146]+train[_][82])*sum((-1)**(a+b+c)*p[8+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A0B0C1)+
               (y_pred[_][147]+train[_][83])*sum((-1)**(a+b+c)*p[16+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A0B1C0)+
               (y_pred[_][148]+train[_][84])*sum((-1)**(a+b+c)*p[16+8+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A0B1C1)+
               (y_pred[_][149]+train[_][85])*sum((-1)**(a+b+c)*p[32+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A1B0C0)+
               (y_pred[_][150]+train[_][86])*sum((-1)**(a+b+c)*p[32+8+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A1B0C1)+
               (y_pred[_][151]+train[_][87])*sum((-1)**(a+b+c)*p[32+16+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A1B1C0)+
               (y_pred[_][152]+train[_][88])*sum((-1)**(a+b+c)*p[32+16+8+a*4+b*2+c] for a in vals for b in vals for c in vals)*np.sum(A1B1C1)+
                 (y_pred[_][153]+train[_][89])*22.))

                 +A0A1*(y_pred[_][64]+train[_][0])+B0B1*(y_pred[_][65]+train[_][1])+C0C1*(y_pred[_][66]+train[_][2])+
               A0A1C1C0*(y_pred[_][67]+train[_][3])+A0A1A0*(y_pred[_][68]+train[_][4])+A1A0A1*(y_pred[_][69]+train[_][5])+
               B0B1B0*(y_pred[_][70]+train[_][6])+B1B0B1*(y_pred[_][71]+train[_][7])+B0B1B0C0*(y_pred[_][72]+train[_][8])+
               C0C1C0*(y_pred[_][73]+train[_][9])+C1C0C1*(y_pred[_][74]+train[_][10])+A0B0B1*(y_pred[_][75]+train[_][11])+
               A1B0B1*(y_pred[_][76]+train[_][12])+A0C0C1*(y_pred[_][77]+train[_][13])+A1C0C1*(y_pred[_][78]+train[_][14])+
               B1B0B1C0*(y_pred[_][79]+train[_][15])+B1B0B1C1*(y_pred[_][80]+train[_][16])+B0C0C1*(y_pred[_][81]+train[_][17])+
               B1C0C1*(y_pred[_][82]+train[_][18])+A0A1B0*(y_pred[_][83]+train[_][19])+A0A1B1*(y_pred[_][84]+train[_][20])+
               A0A1C0*(y_pred[_][85]+train[_][21])+A0A1C1*(y_pred[_][86]+train[_][22])+B0B1C1C0*(y_pred[_][87]+train[_][23])+
               B0B1C0C1*(y_pred[_][88]+train[_][24])+A0C0B0B1*(y_pred[_][89]+train[_][25])+B0B1C0*(y_pred[_][90]+train[_][26])+
               B0B1C1*(y_pred[_][91]+train[_][27])+A0A1A0B0*(y_pred[_][92]+train[_][28])+A0A1A0B1*(y_pred[_][93]+train[_][29])+
               A0A1A0C0*(y_pred[_][94]+train[_][30])+A0A1A0C1*(y_pred[_][95]+train[_][31])+A1A0A1B0*(y_pred[_][96]+train[_][32])+
               A1A0A1B1*(y_pred[_][97]+train[_][33])+A1A0A1C0*(y_pred[_][98]+train[_][34])+A1A0A1C1*(y_pred[_][99]+train[_][35])+
               B0B1B0A0*(y_pred[_][100]+train[_][36])+B0B1B0A1*(y_pred[_][101]+train[_][37])+B0B1B0C1*(y_pred[_][102]+train[_][38])+
               B1B0B1A0*(y_pred[_][103]+train[_][39])+B1B0B1A1*(y_pred[_][104]+train[_][40])+C0C1C0B0*(y_pred[_][105]+train[_][41])+
               C0C1C0B1*(y_pred[_][106]+train[_][42])+C0C1C0A0*(y_pred[_][107]+train[_][43])+C0C1C0A1*(y_pred[_][108]+train[_][44])+
               C1C0C1A0*(y_pred[_][109]+train[_][45])+C1C0C1A1*(y_pred[_][110]+train[_][46])+C1C0C1B0*(y_pred[_][111]+train[_][47])+
               C1C0C1B1*(y_pred[_][112]+train[_][48])+A0B0C0C1*(y_pred[_][113]+train[_][49])+A0B1C0C1*(y_pred[_][114]+train[_][50])+
               A1B0C0C1*(y_pred[_][115]+train[_][51])+A1B1C0C1*(y_pred[_][116]+train[_][52])+A0A1B0B1*(y_pred[_][117]+train[_][53])+
               A0A1C0C1*(y_pred[_][118]+train[_][54])+A0A1B1C0*(y_pred[_][119]+train[_][55])+A0A1B0C0*(y_pred[_][120]+train[_][56])+
               A0A1B0C1*(y_pred[_][121]+train[_][57])+A0A1B1B0*(y_pred[_][122]+train[_][58])+A0A1B1C1*(y_pred[_][123]+train[_][59])+
               A0C1B0B1*(y_pred[_][124]+train[_][60])+A1C1B0B1*(y_pred[_][125]+train[_][61])+A1C0B0B1*(y_pred[_][126]+train[_][62])+
               A0*(y_pred[_][127]+train[_][63])+
               A1*(y_pred[_][128]+train[_][64])+
               B0*(y_pred[_][129]+train[_][65])+
               B1*(y_pred[_][130]+train[_][66])+
               C0*(y_pred[_][131]+train[_][67])+
               C1*(y_pred[_][132]+train[_][68])+
               A0B0*(y_pred[_][133]+train[_][69])+
               A0B1*(y_pred[_][134]+train[_][70])+
               A0C0*(y_pred[_][135]+train[_][71])+
               A0C1*(y_pred[_][136]+train[_][72])+
               A1B0*(y_pred[_][137]+train[_][73])+
               A1B1*(y_pred[_][138]+train[_][74])+
               A1C0*(y_pred[_][139]+train[_][75])+
               A1C1*(y_pred[_][140]+train[_][76])+
               B0C0*(y_pred[_][141]+train[_][77])+
               B0C1*(y_pred[_][142]+train[_][78])+
               B1C0*(y_pred[_][143]+train[_][79])+
               B1C1*(y_pred[_][144]+train[_][80])+
               A0B0C0*(y_pred[_][145]+train[_][81])+
               A0B0C1*(y_pred[_][146]+train[_][82])+
               A0B1C0*(y_pred[_][147]+train[_][83])+
               A0B1C1*(y_pred[_][148]+train[_][84])+
               A1B0C0*(y_pred[_][149]+train[_][85])+
               A1B0C1*(y_pred[_][150]+train[_][86])+
               A1B1C0*(y_pred[_][151]+train[_][87])+
               A1B1C1*(y_pred[_][152]+train[_][88]))
'''
