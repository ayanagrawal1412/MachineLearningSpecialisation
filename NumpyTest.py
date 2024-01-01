import time
import numpy as np
a=np.array([1,2,3,4,5])
print(f"a = {a}")

b=np.array([2,-2,3,0,5])

c=a+b
print(f"c = {c}")

print(f"a-b = {a-b}")

print(a[2:4])

d=np.arange(10)

try:
    print(f"a+d = {a+d}")
except:
    print("the error msg")

print(f"a-d = {a-d}")

print("Hello there")

np.newaxis

c=time.time()

np.zeros(2)

a.reshape