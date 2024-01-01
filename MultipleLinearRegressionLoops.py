import numpy as np
import copy,math
import matplotlib.pyplot as plt

X_train = np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
y_train = np.array([460,232,178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

a=0.01

def compute_model(w,b,X):
    m=X.shape[0]
    f=np.zeros(m)
    for i in range(m):
        f[i]=np.dot(w,X[i])+b
    return f

def compute_cost(w,b,X,y):
    m=X.shape[0]
    J=0
    for i in range(m):
        J+=(np.dot(w,X[i])+b-y[i])**2
    return J/(2*m)

def compute_derivatives(w,b,X,y):
    (m,n)=X.shape
    dJdw=np.zeros(n)
    dJdb=0
    for i in range(m):
        diff = np.dot(w,X[i])+b-y[i]
        for j in range(n):
            dJdw[j]+=diff*X[i,j]
        dJdb+=diff
    return (dJdw/m,dJdb/m)

def gradient_descent(w,b,X,y,a,cost_fn,derivative_fn):
    J=cost_fn(w,b,X,y)
    wl=[]
    bl=[]
    Jl=[]
    n=1
    while True:
        wl.append(w)
        bl.append(b)
        Jl.append(J)
        (dJdw,dJdb)=derivative_fn(w,b,X,y)
        w-=a*dJdw
        b-=a*dJdb
        J=cost_fn(w,b,X,y)
        if J>Jl[-1] or Jl[-1]-J<0.001 or n>1000:
            print(w,b,J,n)
            break
        n+=1
    return (wl,bl,Jl,n)

(w,b,J,n)=gradient_descent(w_init,b_init,X_train,y_train,a,compute_cost,compute_derivatives)

for i in range(n):
    print(f"{J[i]}")