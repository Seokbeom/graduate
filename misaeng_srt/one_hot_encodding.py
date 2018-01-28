
data = open('misaeng_cleansed.txt', 'r', encoding='utf-8')
fulltext= data.read()
fulltext=fulltext.replace('\n',' \n ')
fulltext_list = fulltext.split(' ')
seen = set()
result = [] # every word -> integer  encoding
for i in fulltext_list:
  if i not in seen:
    seen.add(i)
    result.append(i)

'''
data2 = open('misaeng_one_hot_encoded.txt', 'w', encoding='utf-8')
for word in fulltext_list:
    if word !='\n':
        one_hot_encode = str(result.index(word))
        data2.write(one_hot_encode)
        data2.write(' ')
    else:
        data2.write('\n')

print(result[0:100])
print(len(result))
print('\n' in result)

#data2.close()

end = time.time()
print(end-start, '초')
'''