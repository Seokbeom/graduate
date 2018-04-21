from nltk.stem import WordNetLemmatizer
senlen = 5
file_name ='./extracted_data/movie_dialogue_' + str(senlen) +'_T'



nameset=set() # 이름 셋
lines = open('./data/movie_lines.tsv',encoding='utf8').read()[1:].split('\n')
for line in lines:
    _line = line.split('\t')
    if len(_line) >= 5:
        try:
            nameset.add(_line[3][0] + _line[3][1:].lower())
        except:
            pass
nameset.remove('Class')
nameset.remove('All')

wnl = WordNetLemmatizer()
data = open(file_name +'.txt', 'r', encoding='utf8')
whole_text= data.read()
data.close()
whole_text= whole_text.split(' ')
while '' in whole_text:  # 의미없는 '' 제거
    whole_text.remove('')


#등장 빈도가 1개이하면 UNK 로 대체
appeared = set()
selected_word = set()
print('원래 단어 수 : ',len(whole_text))
for index in range(len(whole_text)):
    word = whole_text[index]
    if word in nameset:
        word = 'NAME'
    else:
        word = wnl.lemmatize(whole_text[index]).lower()
    whole_text[index] = word

    if word in appeared:
        selected_word.add(word)
    else:
        appeared.add(word)

words_count = len(selected_word)
QA=('T', 'Q', 'A')
for qa in QA:
    new_file_name = file_name[:-1] + qa
    data = open(new_file_name +'.txt', 'r', encoding='utf8')
    whole_text= data.read()
    whole_text = whole_text.split(' ')
    while '' in whole_text:  # 의미없는 '' 제거
        whole_text.remove('')
    data.close()

    data = open( new_file_name + '_%d.txt'%(words_count),'w')

    for index in range(len(whole_text)):
        word = whole_text[index]
        if word in nameset:
            word = 'NAME'
        else:
            word = wnl.lemmatize(whole_text[index]).lower()

        if word in selected_word:
            data.write(word+" ")
        else:
            data.write("UNK ")

    data.close()

#print(nameset)
print('최종 단어 수 : ',words_count)