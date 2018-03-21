



cleansed = open('cleansed2.txt', 'r', encoding='utf-8')
original = cleansed.read()
new =  original .replace(" -" , "\n")
new =  new.replace("-" , "")
new =  new.replace("  " , " ")
new = new.replace('\n',' <eos>\n ')
output = open('second_cleansed.txt', 'w', encoding='utf-8')

output.write(new)


cleansed.close()
output.close()




