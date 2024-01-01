import numpy as np
import matplotlib.pyplot

x_train = np.array([1.0,2.0])
y_train = np.array([300.0, 500.0])

w0=0
b0=0
a=0.3

def compute_model(w,b,x):
    return w*x+b

def compute_cost(w,b,x,y):
    m=x.shape[0]
    return sum((w*x+b-y)**2)/(2*m)

def compute_derivatives(w,b,x,y):
    m=x.shape[0]
    dj_dw = sum((w*x+b-y)*x)/m
    dj_db = sum(w*x+b-y)/m
    return dj_dw,dj_db

def gradient_descent(w,b,x,y,a,derivative_fn,cost_fn):
    j=cost_fn(w,b,x,y)
    n=1
    wl=[]
    bl=[]
    jl=[]
    while True:
        wl.append(w)
        bl.append(b)
        jl.append(j)
        dj=derivative_fn(w,b,x,y)
        w-=a*dj[0]
        b-=a*dj[1]
        j=cost_fn(w,b,x,y)
        if  len(jl)>400 or j>jl[-1] or jl[-1]-j<0.001:
            print (w,b,j,n)
            break
        n+=1
    return wl,bl,jl,n

(w,b,j,n)=gradient_descent(w0,b0,x_train,y_train,a,compute_derivatives,compute_cost)

for i in range(n):
    print(f"{w[i]:.4f},{b[i]:.4f},{j[i]:.4f}")
