
import re
text='ddddddddddd           dddddddddddddddddddfdDDDDDDdf#\" ++!?!000@#@dfdf'
#text = re.sub(r"[^A-Za-z0-9!?]", " ", text)
text = re.sub("\s+", " ",text)
print(text)


s=" dddddd\n\n ddd\n\n"
print(s.strip())

def f():
    return 1,2

import numpy as np
x=np.array( [[ [1,1,1],[1,1,1],[1,1,1],[1,1,1]],[ [2,2,2],[2,2,2],[2,2,2],[2,2,2]], [ [3,3,3],[3,3,3],[3,3,3],[3,3,3]]])
print(x[0])
xx=np.array([ [0,0,0],[1,1,1],[2,2,2], [3,3,3] ])
print(xx)
x[0]=xx
print(x)
print(xx)
y = np.delete(xx,0,0)
print(y)
x[0] = y