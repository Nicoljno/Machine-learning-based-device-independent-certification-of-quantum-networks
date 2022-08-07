#numerically generation of the input samples used to train the NN 
#used as a device-independent tool for the certification of the
#non-locality of experimental data

from math import *
import numpy as np
import sys


def chsh(P):
    vals=[0,1]
    a=sum((-1)**(a+b)*P[a,b,0,0] for a in vals for b in vals)
    b=sum((-1)**(a+b)*P[a,b,0,1] for a in vals for b in vals)
    c=sum((-1)**(a+b)*P[a,b,1,0] for a in vals for b in vals)
    d=sum((-1)**(a+b)*P[a,b,1,1] for a in vals for b in vals)
    #print(a,b,c,d)
    return a+b+c-d

sys.stdout=open("bell_samples.txt","w")
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
p=np.zeros((2,2,2,2))

#definition of the hoeffding's error for an epsilon = 10**(-5)
errors=np.array([0.0110501283, 0.0110501283, 0.0110501283, 0.0110501283,
                 0.00618567728, 0.00618567728, 0.00618567728, 0.00618567728,
                 0.00745087848, 0.00745087848, 0.00745087848, 0.00745087848,
                 0.00987155205, 0.00987155205, 0.00987155205, 0.00987155205])
#the first exponent defines the epsilon exponent
errors=errors*sqrt(-log(10**(-5)*0.5))/sqrt(-log(10**(-5)*0.5))
errors=np.reshape(errors, (2,2,2,2))
for _ in range(200000):
    v=np.random.uniform(low=0.7, high=1.)
    rho=np.outer(S, S)*v+(1.-v)*np.eye(4)/4.
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    p[a,b,x,y]=np.trace(np.dot(rho, np.kron(A[a*2+x], B[b*2+y])))
    tmp_=np.copy(p)
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    p[a,b,x,y]=p[a,b,x,y]+np.random.normal(loc=0, scale=errors[a,b,x,y])
    tmp=np.copy(p)
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    p[a,b,x,y]=abs(p[a,b,x,y])/sum(abs(tmp[a1,b1,x,y]) for a1 in vals for b1 in vals)
    tmp=np.copy(p)
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    p[a,b,x,y]=tmp[x,y,a,b]
    print(p[0,0,0,0], p[0,0,0,1], p[0,0,1,0], p[0,0,1,1],
          p[0,1,0,0], p[0,1,0,1], p[0,1,1,0], p[0,1,1,1],
          p[1,0,0,0], p[1,0,0,1], p[1,0,1,0], p[1,0,1,1],
          p[1,1,0,0], p[1,1,0,1], p[1,1,1,0], p[1,1,1,1])
    #ceck on no_signaling condition
    no_sign=0
    for a in vals:
        for x in vals:
            no_sign+=abs(sum(p[x,1,a,b]-p[x,0,a,b] for b in vals))
    for b in vals:
        for y in vals:
            no_sign+=abs(sum(p[1,y,a,b]-p[0,y,a,b] for a in vals))
    #print(no_sign)
