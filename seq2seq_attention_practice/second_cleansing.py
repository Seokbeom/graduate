



cleansed = open('test.txt', 'r', encoding='utf-8')
original = cleansed.read()
new =  original .replace(" -" , "\n")
new =  new.replace("-" , "")
new =  new.replace("  " , " ")
#new = new.replace('\n',' <eos>\n ')
new = new.replace('<eos>\n', '<eos>\n ')
output = open('test_cleansed.txt', 'w', encoding='utf-8')

output.write(new)


cleansed.close()
output.close()




