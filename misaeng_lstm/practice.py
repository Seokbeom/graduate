string = '01234567890'

a = string.index('0')
print(a)

a=[0,1,2,3,4,5,1,1]
a=['a','ab','a','a','a',9,'a']
b = a.index('a',2)
c = a.index('a',4)
print(b)
print(c)

import time
a = time.localtime()

print(a[1])
print(a[2])
print(a[3])
print(a[4])
timestring = a[0:4]
print(timestring)
print(type(a))