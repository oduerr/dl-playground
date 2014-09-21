from numpy import *
import numpy as np

d = [[1.0,2],[3,4],[5,6]]
print(type(d)) #List

m = np.asarray(d)
print(str(type(m)) + str(np.shape(m))) #numpy.ndarray 3x2

# Matrix Multiplication
x = np.asarray([1,2])
print(m * x)
print(x * m) #Both OK

x = np.asarray([1,2,3])
print(m * x)
print(x * m) #Both OK
