from nltk.stem import WordNetLemmatizer


file_name ='movie_dialogue_10_Extracted'
wnl = WordNetLemmatizer()
data = open(file_name +'.txt', 'r', encoding='utf8')
whole_text= data.read()
data.close()


nameset=set() # 이름 셋
lines = open('movie_lines.tsv',encoding='utf8').read()[1:].split('\n')
for line in lines:
    _line = line.split('\t')
    if len(_line) >= 5:
        try:
            nameset.add(_line[3][0] + _line[3][1:].lower())
        except:
            pass
nameset.remove('Class')
nameset.remove('All')
whole_text= whole_text.split(' ')
while '' in whole_text:  # 의미없는 '' 제거
    whole_text.remove('')

#등장 빈도가 1개이하면 unknownword 로 대체

appeared = set()
selected_word = set()
data = open(file_name + '_Lemmatized.txt','w')
print('원래 단어 수 : ',len(whole_text))
for index in range(len(whole_text)):
    word = whole_text[index]
    if word in nameset:
        word = 'personname'
    else:
        word = wnl.lemmatize(whole_text[index]).lower()
    whole_text[index] = word
    if word in appeared:
        selected_word.add(word)
    else:
        appeared.add(word)


for index in range(len(whole_text)):
    word = whole_text[index]
    if word in selected_word:
        data.write(word)
        data.write(" ")
    else:
        data.write("unknownword ")

data.close()

print(nameset)
print('최종 단어 수 : ',len(selected_word))