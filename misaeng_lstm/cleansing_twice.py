



cleansed = open('cleansed2.txt', 'r', encoding='utf-8')
original = cleansed.read()
new =  original .replace(" -" , "\n")
new =  new.replace("-" , "")
new =  new.replace("  " , " ")

output = open('cleansed_twice.txt', 'w', encoding='utf-8')

output.write(new)


cleansed.close()
output.close()



