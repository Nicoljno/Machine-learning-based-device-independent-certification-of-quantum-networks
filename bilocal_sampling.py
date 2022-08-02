from math import *
import numpy as np
import sys

sys.stdout=open("bilocal_samples.txt","w")

v0=np.array([1.,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
v1=np.array([0,1.,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
v2=np.array([0,0,1.,0,0,0,0,0,0,0,0,0,0,0,0,0])
v3=np.array([0,0,0,1.,0,0,0,0,0,0,0,0,0,0,0,0])
v4=np.array([0,0,0,0,1.,0,0,0,0,0,0,0,0,0,0,0])
v5=np.array([0,0,0,0,0,1.,0,0,0,0,0,0,0,0,0,0])
v6=np.array([0,0,0,0,0,0,1.,0,0,0,0,0,0,0,0,0])
v7=np.array([0,0,0,0,0,0,0,1.,0,0,0,0,0,0,0,0])
v8=np.array([0,0,0,0,0,0,0,0,1.,0,0,0,0,0,0,0])
v9=np.array([0,0,0,0,0,0,0,0,0,1.,0,0,0,0,0,0])
v10=np.array([0,0,0,0,0,0,0,0,0,0,1.,0,0,0,0,0])
v11=np.array([0,0,0,0,0,0,0,0,0,0,0,1.,0,0,0,0])
v12=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1.,0,0,0])
v13=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1.,0,0])
v14=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.,0])
v15=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.])
sx=np.array([[0,1],[1,0]])
sz=np.array([[1,0],[0,-1]])
Id=np.array([[1,0],[0,1]])
vals=[0,1]
S=(-v6+v5+v10-v9)/2.
one=np.array([0,1])
zero=np.array([1,0])
proj=[]
proj.append(zero)
proj.append(one)
proj.append(cos(pi/4)*zero+sin(pi/4)*one)
proj.append(cos(pi/4)*one-sin(pi/4)*zero)
proj.append(cos(pi/8)*zero+sin(pi/8)*one)
proj.append(cos(pi/8)*one-sin(pi/8)*zero)
proj.append(cos(3*pi/8)*zero+sin(3*pi/8)*one)
proj.append(cos(3*pi/8)*one-sin(3*pi/8)*zero)
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
pb=np.zeros((2,2,2,2))
P_=np.zeros((2,2,2,2,2,2,2))
p=np.zeros((2,2,2,2,2,2))

for _ in range(100000):
    v=np.random.uniform(low=0.5, high=1.)
    rho=v*np.outer(S, S)*1+(1.-v)*np.eye(16)/16.
    #print(rho)
    for a in vals:
        for b in vals:
            for b1 in vals:
                for c in vals:
                    for x in vals:
                        for y in vals:
                            for z in vals:
                                P_[a,b,b1,c,x,y,z]=np.trace(np.dot(rho, np.kron(np.kron(A[a*2+x], B[b*2+y]), np.kron(B[b1*2+y], C[c*2+z]))))
    for a in vals:
        for c in vals:
            for x in vals:
                for y in vals:
                    for z in vals:
                        p[a,0,c,x,y,z]=P_[a,0,1,c,x,y,z]+P_[a,1,0,c,x,y,z]
                        p[a,1,c,x,y,z]=P_[a,0,0,c,x,y,z]+P_[a,1,1,c,x,y,z]
    #print(sum(p[a,b,c,0,0,0] for a in vals for b in vals for c in vals))

    #adding poisson distributed errors to the data, the number of coincidences 
    
    tmp_=np.copy(p)
    for a in vals:
        for b in vals:
            for c in vals:
                for x in vals:
                    for y in vals:
                        for z in vals:
                            p[a,b,c,x,y,z]=np.random.poisson(tmp_[a,b,c,x,y,z]*200000., 1)/200000.
    #print(p[0,0,0,0,0,0]+p[0,0,1,0,0,0]+p[0,1,0,0,0,0]+p[0,1,1,0,0,0]+p[1,0,0,0,0,0]+p[1,0,1,0,0,0]+p[1,1,0,0,0,0]+p[1,1,1,0,0,0])
    tmp=np.copy(p)
    for a in vals:
        for b in vals:
            for c in vals:
                for x in vals:
                    for y in vals:
                        for z in vals:
                            p[a,b,c,x,y,z]=abs(p[a,b,c,x,y,z])/sum(abs(tmp[a1,b1,c1,x,y,z]) for a1 in vals for b1 in vals for c1 in vals)

    #we need our probability distribution to be of the form p(x,y,z|a,b,c) to fetch it as inputs to the NN
    
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



'''
v0=np.array([1.,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
v1=np.array([0,1.,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
v2=np.array([0,0,1.,0,0,0,0,0,0,0,0,0,0,0,0,0])
v3=np.array([0,0,0,1.,0,0,0,0,0,0,0,0,0,0,0,0])
v4=np.array([0,0,0,0,1.,0,0,0,0,0,0,0,0,0,0,0])
v5=np.array([0,0,0,0,0,1.,0,0,0,0,0,0,0,0,0,0])
v6=np.array([0,0,0,0,0,0,1.,0,0,0,0,0,0,0,0,0])
v7=np.array([0,0,0,0,0,0,0,1.,0,0,0,0,0,0,0,0])
v8=np.array([0,0,0,0,0,0,0,0,1.,0,0,0,0,0,0,0])
v9=np.array([0,0,0,0,0,0,0,0,0,1.,0,0,0,0,0,0])
v10=np.array([0,0,0,0,0,0,0,0,0,0,1.,0,0,0,0,0])
v11=np.array([0,0,0,0,0,0,0,0,0,0,0,1.,0,0,0,0])
v12=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1.,0,0,0])
v13=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1.,0,0])
v14=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.,0])
v15=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.])
sx=np.array([[0,1],[1,0]])
sz=np.array([[1,0],[0,-1]])
Id=np.array([[1,0],[0,1]])
vals=[0,1]
S=(-v6+v5+v10-v9)/2.
one=np.array([0,1])
zero=np.array([1,0])
proj=[]
proj.append(zero)
proj.append(one)
proj.append(cos(pi/4)*zero+sin(pi/4)*one)
proj.append(cos(pi/4)*one-sin(pi/4)*zero)
proj.append(cos(pi/8)*zero+sin(pi/8)*one)
proj.append(cos(pi/8)*one-sin(pi/8)*zero)
proj.append(cos(3*pi/8)*zero+sin(3*pi/8)*one)
proj.append(cos(3*pi/8)*one-sin(3*pi/8)*zero)
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
pb=np.zeros((2,2,2,2))
P_=np.zeros((2,2,2,2,2,2,2))
p=np.zeros((2,2,2,2,2,2))

for _ in range(100000):
    v=np.random.uniform(low=0.5, high=1.)
    rho=np.outer(S, S)*v+(1.-v)*np.eye(16)/16.
    #print(rho)
    for a in vals:
        for b in vals:
            for b1 in vals:
                for c in vals:
                    for x in vals:
                        for y in vals:
                            for z in vals:
                                P_[a,b,b1,c,x,y,z]=np.trace(np.dot(rho, np.kron(np.kron(A[a*2+x], B[b*2+y]), np.kron(B[b1*2+y], C[c*2+z]))))
    for a in vals:
        for c in vals:
            for x in vals:
                for y in vals:
                    for z in vals:
                        p[a,0,c,x,y,z]=P_[a,0,1,c,x,y,z]+P_[a,1,0,c,x,y,z]
                        p[a,1,c,x,y,z]=P_[a,0,0,c,x,y,z]+P_[a,1,1,c,x,y,z]
    #print(sum(p[a,b,c,0,0,0] for a in vals for b in vals for c in vals))
    tmp_=np.copy(p)
    for a in vals:
        for b in vals:
            for c in vals:
                for x in vals:
                    for y in vals:
                        for z in vals:
                            p[a,b,c,x,y,z]=np.random.poisson(tmp_[a,b,c,x,y,z]*1000., 1)/1000.
    #print(p[0,0,0,0,0,0]+p[0,0,1,0,0,0]+p[0,1,0,0,0,0]+p[0,1,1,0,0,0]+p[1,0,0,0,0,0]+p[1,0,1,0,0,0]+p[1,1,0,0,0,0]+p[1,1,1,0,0,0])
    tmp=np.copy(p)
    for a in vals:
        for b in vals:
            for c in vals:
                for x in vals:
                    for y in vals:
                        for z in vals:
                            p[a,b,c,x,y,z]=abs(p[a,b,c,x,y,z])/sum(abs(tmp[a1,b1,c1,x,y,z]) for a1 in vals for b1 in vals for c1 in vals)
    tmp=np.copy(p)
    for a in vals:
        for b in vals:
            for c in vals:
                for x in vals:
                    for y in vals:
                        for z in vals:
                            p[a,b,c,x,y,z]=tmp[x,y,z,a,b,c]
#    print(p[0,0,0,0,0,0], p[0,0,0,0,0,1], p[0,0,0,0,1,0], p[0,0,0,0,1,1], p[0,0,0,1,0,0], p[0,0,0,1,0,1], p[0,0,0,1,1,0], p[0,0,0,1,1,1],
#          p[0,0,1,0,0,0], p[0,0,1,0,0,1], p[0,0,0,0,1,0], p[0,0,1,0,1,1], p[0,0,1,1,0,0], p[0,0,1,1,0,1], p[0,0,1,1,1,0], p[0,0,1,1,1,1],
#          p[0,1,0,0,0,0], p[0,1,0,0,0,1], p[0,1,0,0,1,0], p[0,1,0,0,1,1], p[0,1,0,1,0,0], p[0,1,0,1,0,1], p[0,1,0,1,1,0], p[0,1,0,1,1,1],
#          p[0,1,1,0,0,0], p[0,1,1,0,0,1], p[0,1,1,0,1,0], p[0,1,1,0,1,1], p[0,1,1,1,0,0], p[0,1,1,1,0,1], p[0,1,1,1,1,0], p[0,1,1,1,1,1],
#          p[1,0,0,0,0,0], p[1,0,0,0,0,1], p[1,0,0,0,1,0], p[1,0,0,0,1,1], p[1,0,0,1,0,0], p[1,0,0,1,0,1], p[1,0,0,1,1,0], p[1,0,0,1,1,1],
#          p[1,0,1,0,0,0], p[1,0,1,0,0,1], p[1,0,1,0,1,0], p[1,0,1,0,1,1], p[1,0,1,1,0,0], p[1,0,1,1,0,1], p[1,0,1,1,1,0], p[1,0,1,1,1,1],
#          p[1,1,0,0,0,0], p[1,1,0,0,0,1], p[1,1,0,0,1,0], p[1,1,0,0,1,1], p[1,1,0,1,0,0], p[1,1,0,1,0,1], p[1,1,0,1,1,0], p[1,1,0,1,1,1],
#          p[1,1,1,0,0,0], p[1,1,1,0,0,1], p[1,1,1,0,1,0], p[1,1,1,0,1,1], p[1,1,1,1,0,0], p[1,1,1,1,0,1], p[1,1,1,1,1,0], p[1,1,1,1,1,1])
    no_sign=0
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    no_sign+=abs(sum(p[x,y,1,a,b,c]-p[x,y,0,a,b,c] for c in vals))
    for a in vals:
        for c in vals:
            for x in vals:
                for z in vals:
                    no_sign+=abs(sum(p[x,1,z,a,b,c]-p[x,0,z,a,b,c] for b in vals))
    for b in vals:
        for c in vals:
            for y in vals:
                for z in vals:
                    no_sign+=abs(sum(p[1,y,z,a,b,c]-p[0,y,z,a,b,c] for a in vals))
    print(no_sign)
'''

'''
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
for _ in range(100000):
    v=np.random.uniform(low=0.7, high=1.)
    v1=np.random.uniform(low=0.7, high=1.)
    rho=np.kron(np.outer(S, S)*v+(1.-v)*np.eye(4)/4.,np.outer(S, S)*v1+(1.-v1)*np.eye(4)/4.)
    #print(rho)
    for a in vals:
        for b in vals:
            for b1 in vals:
                for c in vals:
                    for x in vals:
                        for y in vals:
                            for z in vals:
                                P_[a,b,b1,c,x,y,z]=np.trace(np.dot(rho, np.kron(np.kron(A[a*2+x], B[b*2+y]), np.kron(B[b1*2+y], C[c*2+z]))))
    for a in vals:
        for c in vals:
            for x in vals:
                for y in vals:
                    for z in vals:
                        p[a,0,c,x,y,z]=P_[a,0,1,c,x,y,z]+P_[a,1,0,c,x,y,z]
                        p[a,1,c,x,y,z]=P_[a,0,0,c,x,y,z]+P_[a,1,1,c,x,y,z]
    #print(sum(p[a,b,c,0,0,0] for a in vals for b in vals for c in vals))
    tmp_=np.copy(p)
    for a in vals:
        for b in vals:
            for c in vals:
                for x in vals:
                    for y in vals:
                        for z in vals:
                            p[a,b,c,x,y,z]=np.random.poisson(tmp_[a,b,c,x,y,z]*1000., 1)/1000.
    #print(p[0,0,0,0,0,0]+p[0,0,1,0,0,0]+p[0,1,0,0,0,0]+p[0,1,1,0,0,0]+p[1,0,0,0,0,0]+p[1,0,1,0,0,0]+p[1,1,0,0,0,0]+p[1,1,1,0,0,0])
    tmp=np.copy(p)
    for a in vals:
        for b in vals:
            for c in vals:
                for x in vals:
                    for y in vals:
                        for z in vals:
                            p[a,b,c,x,y,z]=abs(p[a,b,c,x,y,z])/sum(abs(tmp[a1,b1,c1,x,y,z]) for a1 in vals for b1 in vals for c1 in vals)
    tmp=np.copy(p)
    for a in vals:
        for b in vals:
            for c in vals:
                for x in vals:
                    for y in vals:
                        for z in vals:
                            p[a,b,c,x,y,z]=tmp[x,y,z,a,b,c]
#    print(p[0,0,0,0,0,0], p[0,0,0,0,0,1], p[0,0,0,0,1,0], p[0,0,0,0,1,1], p[0,0,0,1,0,0], p[0,0,0,1,0,1], p[0,0,0,1,1,0], p[0,0,0,1,1,1],
#          p[0,0,1,0,0,0], p[0,0,1,0,0,1], p[0,0,0,0,1,0], p[0,0,1,0,1,1], p[0,0,1,1,0,0], p[0,0,1,1,0,1], p[0,0,1,1,1,0], p[0,0,1,1,1,1],
#          p[0,1,0,0,0,0], p[0,1,0,0,0,1], p[0,1,0,0,1,0], p[0,1,0,0,1,1], p[0,1,0,1,0,0], p[0,1,0,1,0,1], p[0,1,0,1,1,0], p[0,1,0,1,1,1],
#          p[0,1,1,0,0,0], p[0,1,1,0,0,1], p[0,1,1,0,1,0], p[0,1,1,0,1,1], p[0,1,1,1,0,0], p[0,1,1,1,0,1], p[0,1,1,1,1,0], p[0,1,1,1,1,1],
#          p[1,0,0,0,0,0], p[1,0,0,0,0,1], p[1,0,0,0,1,0], p[1,0,0,0,1,1], p[1,0,0,1,0,0], p[1,0,0,1,0,1], p[1,0,0,1,1,0], p[1,0,0,1,1,1],
#          p[1,0,1,0,0,0], p[1,0,1,0,0,1], p[1,0,1,0,1,0], p[1,0,1,0,1,1], p[1,0,1,1,0,0], p[1,0,1,1,0,1], p[1,0,1,1,1,0], p[1,0,1,1,1,1],
#          p[1,1,0,0,0,0], p[1,1,0,0,0,1], p[1,1,0,0,1,0], p[1,1,0,0,1,1], p[1,1,0,1,0,0], p[1,1,0,1,0,1], p[1,1,0,1,1,0], p[1,1,0,1,1,1],
#          p[1,1,1,0,0,0], p[1,1,1,0,0,1], p[1,1,1,0,1,0], p[1,1,1,0,1,1], p[1,1,1,1,0,0], p[1,1,1,1,0,1], p[1,1,1,1,1,0], p[1,1,1,1,1,1])
    no_sign=0
    for a in vals:
        for b in vals:
            for x in vals:
                for y in vals:
                    no_sign+=abs(sum(p[x,y,1,a,b,c]-p[x,y,0,a,b,c] for c in vals))
    for a in vals:
        for c in vals:
            for x in vals:
                for z in vals:
                    no_sign+=abs(sum(p[x,1,z,a,b,c]-p[x,0,z,a,b,c] for b in vals))
    for b in vals:
        for c in vals:
            for y in vals:
                for z in vals:
                    no_sign+=abs(sum(p[1,y,z,a,b,c]-p[0,y,z,a,b,c] for a in vals))
    print(no_sign)
'''
