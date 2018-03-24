
input='movie_dialogue.txt'
import re

data= open(input, 'r', encoding='utf-8')
output = open('cleansed_' + input, 'w', encoding='utf-8')

while 1:
    line = data.readline()
    #print(line)
    if not line:
        print('finished')
        break
    if len(line) < 1:
        print('len(line) is 0')
        break
    line = line.lower()
    line = line.replace("\'re", " are ").replace("\'ll", " will ").replace("\'m", " am ").replace("?", " ? ").replace("!", " ! ")
    line = re.sub(r"[^a-z0-9!?]", " ", line)
    line = re.sub("\s+", " ",line) # 연속되는 공백문자 공백하나로 통일
    line = line.strip()
    output.writelines(line)
    output.write(" <eos>\n ")


output.close()





