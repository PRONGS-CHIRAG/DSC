import numpy as np 
import torch as t 
import helper
x=t.rand(4,4)
print(x)
y=t.ones(x.size())
print(y)
z=x+y
print(z)
print(z[0])
print(z[2:,2:])
z.add(1)
print(z)
z.add_(1)
print(z)
print(z.size())
print(z.resize_(2,2))
a=np.random.rand(4,4)
print(a)
b=t.from_numpy(a)
print(b)
c=b.numpy()
print(c)
print(b.mul_(3))