import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array ([300.0, 500.0])

w=50
b=50

#a = np.zeros(3)
#print (a)

def compute_model_output(w,b,x):
    m=len(x)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i]+b
    return f_wb

y_hat=compute_model_output(w,b,x_train)
#print(y_hat)
#j=0.0
def compute_cost(w,b,x,y):
    m=len(x)
    j=0
    for i in range(m):
        j+=(y_hat[i]-y[i])**2
    return j/(2*m)

j_wb=compute_cost(w,b,x_train,y_train)

print (j_wb)

dj_dw = 0
dj_db = 0
m=len(x_train)
for i in range(m):
    dj_dw += (w*x_train[i]+b-y_train[i])*x_train[i]/m
    dj_db += (w*x_train[i]+b-y_train[i])/m

w-=dj_dw*0.1
b-=dj_db*0.1

print(f"dj_dw is {dj_dw} and dj_db is {dj_db}")
print (f"w is {w} and b is {b}")

print (j_wb)
print (dj_dw)
print (dj_db)





#tst= ((y_hat,j_wb))
#print (tst[0][0])

f_wb = np.zeros(1)
j_wb = 0.0
def compute_model_output_n_cost(w,b,x,y,f_wb,j_wb):
    m=len(x)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i]+b