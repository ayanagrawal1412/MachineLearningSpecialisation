import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

#print(x_train.shape)

#m = x_train.shape[0]
#print (m)

print (len(x_train))

#i=1
#x_i = x_train[i]
#y_i = y_train[i]
#print (f"value of x_{i} is {x_i}")
#print (f"value of y_{i} is {y_i}")

#plt.scatter(x_train,y_train,marker='x',c='r')
#plt.show()

w=200
b=100

#f_wb = np.zeros(m)
#print (f_wb)

def compute_model_output(x,w,b):
    m=len(x)
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i]+b
    return f_wb

#print (compute_model_output(x_train,w,b))

tmp_f_wb = compute_model_output(x_train,w,b)

plt.plot(x_train,tmp_f_wb,c='b',label='Our prediction')
plt.scatter(x_train,y_train,marker='x',c='r',label='Actual Values')

plt.title('Housing Prices')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.legend()
plt.show()

x=1.2
cost=w*x+b
print(f"${cost:.0f} thousand dollars")