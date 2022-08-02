import numpy as np
from math import *
import pandas as pd
import sys


input_=pd.read_table('bell_samples.txt', delimiter=" ", header=None)
input_=input_.to_numpy()
input_=input_.astype('float32')
p=np.copy(input_)
vals = [0,1]
sys.stdout=open('bonet_samples.txt','w')
p_bon=np.ones((2,2,3,2))
for _ in range(100000):
    for a in vals:
        for b in vals:
            for y in vals:
                p_bon[a,b,2,y]=abs(a-1)*sum(p[0][y*4+a_*2+b] for a_ in vals)
    print(p[_][0], p[_][1], p[_][2], p[_][3],
          p[_][4], p[_][5], p[_][6], p[_][7],
          p[_][8], p[_][9], p[_][10], p[_][11],
          p[_][12], p[_][13], p[_][14], p[_][15],
          p_bon[0,0,2,0], p_bon[0,1,2,0], p_bon[1,0,2,0], p_bon[1,1,2,0],
          p_bon[0,0,2,1], p_bon[0,1,2,1], p_bon[1,0,2,1], p_bon[1,1,2,1])
