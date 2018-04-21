from sklearn.preprocessing import normalize
import numpy as np

a= np.array([0.1,0.2,-0.3], dtype=np.float)
a=a.reshape(1, -1)
b= normalize(a)
print(a)
print(b)
