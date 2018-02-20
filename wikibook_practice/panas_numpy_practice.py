import pandas as pd
import numpy as np

a = pd.DataFrame([
    [1,2,3],
    [4,5,6],
    [7,8,9]


])
print(a)

s= pd.Series([1,2,3])
print(s)

a= pd.DataFrame({
    "row1": [1,2,3,4],
    "row2":[4,5,6,7]

})

print(a[["row1","row2"]])

a=a.as_matrix()
print(type(a))
print(a)