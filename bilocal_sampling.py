#numerically generation of the input samples used to train the NN 
#used as a device-independent tool for the certification of the
#non-locality of experimental data for the bilocal scenario

from math import *
import numpy as np
import sys

sys.stdout=open("bilocal_samples.txt","w")


v0=np.array([1.,0,0,0])
v1=np.array([0,1.,0,0])
v2=np.array([0,0,1.,0])
v3=np.array([0,0,0,1.])
sx=np.array([[0,1],[1,0]])
sz=np.array([[1,0],[0,-1]])
Id=np.array([[1,0],[0,1]])
vals=[0,1]
S=(v1-v2)/sqrt(2)
one=np.array([0,1])
zero=np.array([1,0])

#projectors generation

proj=[]
proj.append(zero)
proj.append(one)
proj.append((zero+one)/sqrt(2))
proj.append((zero-one)/sqrt(2))
proj.append(cos(pi/8)*zero+sin(pi/8)*one)
proj.append(cos(pi/8)*one-sin(pi/8)*zero)
proj.append(cos(pi/8)*zero-sin(pi/8)*one)
proj.append(cos(pi/8)*one+sin(pi/8)*zero)
A=[]
B=[]
A.append(np.outer(proj[0], proj[0]))
A.append(np.outer(proj[2], proj[2]))
A.append(np.outer(proj[1], proj[1]))
A.append(np.outer(proj[3], proj[3]))
C=A
B.append(np.outer(proj[4], proj[4]))
B.append(np.outer(proj[6], proj[6]))
B.append(np.outer(proj[5], proj[5]))
B.append(np.outer(proj[7], proj[7]))

P_=np.zeros((2,2,2,2,2,2,2))
p=np.zeros((2,2,2,2,2,2))
pb=np.zeros((2,2,2,2))
pb1=np.zeros((2,2,2,2))

#definition of the hoeffding's error for an epsilon = 10**(-5)
errors=np.array([0.0110501283, 0.0110501283, 0.0110501283, 0.0110501283,
                 0.00618567728, 0.00618567728, 0.00618567728, 0.00618567728,
                 0.00745087848, 0.00745087848, 0.00745087848, 0.00745087848,
                 0.00987155205, 0.00987155205, 0.00987155205, 0.00987155205])
#the first exponent defines the epsilon exponent

for _ in range(100000):
    v=np.random.uniform(low=0.7, high=1.)
    v1=np.random.uniform(low=0.7, high=1.)
    rho=np.outer(S, S)*v+(1.-v)*np.eye(4)/4.
    rho1=np.outer(S, S)*v1+(1.-v1)*np.eye(4)/4.
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    pb[a,b,x,y]=np.trace(np.dot(rho, np.kron(A[a*2+x], B[b*2+y])))
    tmp_=np.copy(pb)
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    pb[a,b,x,y]=pb[a,b,x,y]+np.random.normal(loc=0, scale=errors[a,b,x,y])
    tmp=np.copy(pb)
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    pb[a,b,x,y]=abs(pb[a,b,x,y])/sum(abs(tmp[a1,b1,x,y]) for a1 in vals for b1 in vals)
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    pb1[a,b,x,y]=np.trace(np.dot(rho1, np.kron(A[a*2+x], B[b*2+y])))
    tmp_=np.copy(pb1)
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    pb1[a,b,x,y]=pb1[a,b,x,y]+np.random.normal(loc=0, scale=errors[a,b,x,y])
    tmp=np.copy(pb1)
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    pb1[a,b,x,y]=abs(pb1[a,b,x,y])/sum(abs(tmp[a1,b1,x,y]) for a1 in vals for b1 in vals)
    for a in vals:
        for b in vals:
            for b1 in vals:
                for c in vals:
                    for x in vals:
                        for y in vals:
                            for z in vals:
                                P_[a,b,b1,c,x,y,z]=pb[a,b,x,y]*pb1[b1,c,y,z]
    for a in vals:
        for c in vals:
            for x in vals:
                for y in vals:
                    for z in vals:
                        p[a,0,c,x,y,z]=P_[a,0,1,c,x,y,z]+P_[a,1,0,c,x,y,z]
                        p[a,1,c,x,y,z]=P_[a,0,0,c,x,y,z]+P_[a,1,1,c,x,y,z]
    #print(sum(p[a,b,c,0,0,0] for a in vals for b in vals for c in vals))
    tmp=np.copy(p)
    for a in vals:
        for b in vals:
            for c in vals:
                for x in vals:
                    for y in vals:
                        for z in vals:
                            p[a,b,c,x,y,z]=tmp[x,y,z,a,b,c]
    print(p[0,0,0,0,0,0], p[0,0,0,0,0,1], p[0,0,0,0,1,0], p[0,0,0,0,1,1], p[0,0,0,1,0,0], p[0,0,0,1,0,1], p[0,0,0,1,1,0], p[0,0,0,1,1,1],
          p[0,0,1,0,0,0], p[0,0,1,0,0,1], p[0,0,0,0,1,0], p[0,0,1,0,1,1], p[0,0,1,1,0,0], p[0,0,1,1,0,1], p[0,0,1,1,1,0], p[0,0,1,1,1,1],
          p[0,1,0,0,0,0], p[0,1,0,0,0,1], p[0,1,0,0,1,0], p[0,1,0,0,1,1], p[0,1,0,1,0,0], p[0,1,0,1,0,1], p[0,1,0,1,1,0], p[0,1,0,1,1,1],
          p[0,1,1,0,0,0], p[0,1,1,0,0,1], p[0,1,1,0,1,0], p[0,1,1,0,1,1], p[0,1,1,1,0,0], p[0,1,1,1,0,1], p[0,1,1,1,1,0], p[0,1,1,1,1,1],
          p[1,0,0,0,0,0], p[1,0,0,0,0,1], p[1,0,0,0,1,0], p[1,0,0,0,1,1], p[1,0,0,1,0,0], p[1,0,0,1,0,1], p[1,0,0,1,1,0], p[1,0,0,1,1,1],
          p[1,0,1,0,0,0], p[1,0,1,0,0,1], p[1,0,1,0,1,0], p[1,0,1,0,1,1], p[1,0,1,1,0,0], p[1,0,1,1,0,1], p[1,0,1,1,1,0], p[1,0,1,1,1,1],
          p[1,1,0,0,0,0], p[1,1,0,0,0,1], p[1,1,0,0,1,0], p[1,1,0,0,1,1], p[1,1,0,1,0,0], p[1,1,0,1,0,1], p[1,1,0,1,1,0], p[1,1,0,1,1,1],
          p[1,1,1,0,0,0], p[1,1,1,0,0,1], p[1,1,1,0,1,0], p[1,1,1,0,1,1], p[1,1,1,1,0,0], p[1,1,1,1,0,1], p[1,1,1,1,1,0], p[1,1,1,1,1,1])
