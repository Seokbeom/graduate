
file_name ='cleansed_movie_dialogue'
data = open(file_name +'.txt', 'r', encoding='utf8')
whole_text= data.read()
data.close()

#whole_text = whole_text.replace('\n', ' \n ')

whole_text= whole_text.split(' ')
while '' in whole_text:  # 의미없는 '' 제거
    whole_text.remove('')

#등장 빈도가 1개이하면 unknownword 로 대체
#appeared_count=[0 for word in whole_text ]
appeared = set()
selected_word = set()


data = open(file_name + '_shrinked.txt','w')
print('원래 단어 수 : ',len(whole_text))
for index in range(len(whole_text)):
    word = whole_text[index]
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