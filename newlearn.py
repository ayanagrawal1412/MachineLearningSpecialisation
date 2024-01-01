import numpy as np

a=np.array([[1,2,3],[3,5,8]])

print(a)

a.sum()

np.sum(a,0)
b=1234.56
print(f"{b:6.3e}")

np.sum(a,-1)

X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)
d=np.dot(X,w)
print(d)
Y= np.array([[1,2,3,4]])
v=np.array([2,2,2,2])
e=np.dot(Y,v)
print(e.shape)
print(e.shape[0])
print(e[0].shape)

a=np.arange(12).reshape(-1,3)
b=np.array([2,3,5])
np.dot(a,b)

np.dot(b,a)

b=np.arange(1,13).reshape(2,3,2)
np.dot(a,b).shape

n=[1,2,3]
for a in n:
    print(a)
