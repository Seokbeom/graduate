list1=[0,1,2,3,4,5,6,'d']

for a ,b in enumerate(list1):
    print(a, b)
a= list1[:-1]
print(a)
a.remove(1)
print(a)

a= dict()
a.update()
print(len(a))
print(a)

a='dddd'
a += 'ccc'
print(a)