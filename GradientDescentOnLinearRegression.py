import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0,2.0])
y_train = np.array([300.0, 500.0])

w=199
b=101

def compute_cost(w,b,x,y):
    m=len(x)
    cost = 0
    for i in range(m):
        y_hat = w*x[i]+b
        cost +=(y_hat-y[i])**2
    return cost/(2*m)

def compute_model(w,b,x,y):
    m=len(x)
    f=np.zeros(m)
    for i in range(m):
        f[i]=w*x[i]+b
    return f
"""
print(f"w = {w}")
print(f"b = {b}")
print(f"Prediction is {compute_model(w,b,x_train,y_train)}")
print(f"Cost is {compute_cost(w,b,x_train,y_train)}")
"""
def compute_derivatives(w,b,x,y):
    m=len(x)
    der=np.zeros(2)
    for i in range(m):
        der[0]+=(w*x[i]+b-y[i])*x[i]/m
        der[1]+=(w*x[i]+b-y[i])/m
    return der

def compute_best_fit(x,y,w,b,a,s):
    cost=compute_cost(w,b,x,y)
    cost_prev=cost+s+1
    #print (cost_prev)
    #print(cost)
    while cost_prev>cost and cost_prev-cost>s:
        cost_prev=cost
        #cost=compute_cost(w,b,x,y)
        der=compute_derivatives(w,b,x,y)
        print(f"w = {w}")
        print(f"b = {b}")
        print(f"Cost is {cost}")
        print(f"dJ_dw={der[0]}")
        print(f"dJ_db={der[1]}")
        w-=a*der[0]
        b-=a*der[1]
        cost = compute_cost(w,b,x,y)
    print (cost_prev)
    print(cost)
    print(w)
    print(b)

compute_best_fit(x_train,y_train,w,b,0.5,0.0000001)

"""
plt.plot(x_train,compute_model(w,b,x_train,y_train),c='b',label='Our prediction')
plt.scatter(x_train,y_train,marker='x',c='r',label='Actual Values')
plt.title('Housing Prices')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.legend()
plt.show()
"""