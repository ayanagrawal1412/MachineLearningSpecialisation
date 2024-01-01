import numpy as np

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

w0=0
b0=0

a=0.3

def compute_model(w,b,x):
    m=x.shape[0]
    f=np.zeros(m)
    for i in range(m):
        f[i]=w*x[i]+b
    return f

def compute_cost(w,b,x,y):
    m=x.shape[0]
    J=0
    for i in range(m):
        J+=(w*x[i]+b-y[i])**2
    J/=2*m
    return J

def compute_derivatives(w,b,x,y):
    m=x.shape[0]
    dJdw=0
    dJdb=0
    for i in range(m):
        er = w*x[i]+b-y[i]
        dJdw+=er*x[i]
        dJdb+=er
    return (dJdw/m,dJdb/m)

def gradient_descent(w,b,x,y,cost_fn,derivative_fn,a):
    J=cost_fn(w,b,x,y)
    wl=[]
    bl=[]
    Jl=[]
    n=1
    while True:
        wl.append(w)
        bl.append(b)
        Jl.append(J)
        (dJdw,dJdb)=derivative_fn(w,b,x,y)
        w-=a*dJdw
        b-=a*dJdb
        J=cost_fn(w,b,x,y)
        if J>Jl[-1] or Jl[-1]-J<0.001:
            print(w,b,J,n)
            break
        n+=1
    return (wl,bl,Jl,n)

(w,b,J,n)=gradient_descent(w0,b0,x_train,y_train,compute_cost,compute_derivatives,a)

for i in range(n):
    print(f"{w[i]:.4f},{b[i]:.4f},{J[i]:.4f}")
