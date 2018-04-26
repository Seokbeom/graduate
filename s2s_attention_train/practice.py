from sklearn.preprocessing import normalize
import numpy as np

a= np.array([0.1,0.2,-0.3], dtype=np.float)
a=a.reshape(1, -1)
b= normalize(a)
print(a)
print(b)

a=[0,1,2,3,4,5,6,7,8,9]
a=[ i for i in range(100)]
b=[ i for i in range(100)]
seed = 7

print(seed)
#print(np.random.seed(seed))
print(seed)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.2, random_state=7)
print(X_train)
print(X_test)
