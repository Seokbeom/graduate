import numpy as np


data = open('cleansed.txt', 'r', encoding='utf-8')
text= data.read()
temp_text = text.split(' ')
while '' in temp_text:  # 의미없는 '' 제거
    temp_text.remove('')
text = ' '.join (temp_text)
text=text.replace('\n',' \n ') # ' \n ' 야예 뺄지 고민

text = text.split(" ")
print('코퍼스의 길이: ', len(text))
#print(text)


#문자를 하나하나 읽어 들이고 ID 붙이기
chars =sorted(list(set(text)))
#chars =set(text)
print('사용되는 문자수 : ', len(chars))
#print(chars)
char_indices = dict((c,i) for i, c in enumerate(chars))
indices_char = dict((i ,c ) for i, c in enumerate(chars))

#텍스트를 maxlen 개의 문자로 자르고 다음에 오는 문자 등록하기
maxlen = 20  # 한 문장을 20 개의 단어로 봄
step =3 # 3개 단어 단위로 건너뚜면서 새로운 문장을 정의
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print("학습할 구문의 수: ", len(sentences))
print("텍스트를 ID 백터로 변환합니다 ")
X= np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y=np.zeros((len(sentences), len(chars)), dtype = np.bool)


for i,sentence in enumerate(sentences): # 명확한 이해 필요
    #while '' in sentence : # 의미없는 '' 제거
        #sentence.remove('')
    print(i, sentence)
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


probas = np.random.multinomial(2,[0.3, 0.3, 0.3],size = 10)
print(probas)
list1= ['a', 'b','c']
a = ' '.join(list1)
print(a)