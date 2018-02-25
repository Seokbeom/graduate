#p321
#과적합이 발생하는듯  같은 단어가 계속 반복됨
# word2vec으로 단어를 백터화 해야 하지 않을까 지금은 id로 그냥 변환했지만
# 맨 처음 문장 사용자가 넣게끔
# 데이터 더 많이
# 뉴럴넷 최적화
# 엔터키 여부


import time
start = time.time()
import codecs  #?

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop # what is it

import numpy as np
import random, sys # sys 알기



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
    #print(i, sentence)
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

#모델 구축

print("모델  구축")
model = Sequential()
model.add(LSTM(128, input_shape = (maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss= 'categorical_crossentropy', optimizer = optimizer)


#후보를 배열에서 꺼내기
def sample(preds, temperature = 1):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, size = 1) # preds의 확률대로 주사위를 1번 던저서 나온 결과를 출력, size 번 반복
    return np.argmax(probas)

for iteration in range(1,60):
    print()
    print('----------------------------')
    print('반복 = ', iteration)
    model.fit(X,y,batch_size=128, epochs=1) # ???이해하기
    #임의의 시작 텍스트 .. 이걸 사용자가 인풋으로 줄 수 있게 해야 할듯
    start_index = random.randint(0, len(text) - maxlen -1 )
    #다양성은 생략
    for diversity in [1.0]: #0.2 ~1.2
        generated = ''
        sentence_list = text[start_index : start_index + maxlen]
        print('시드 : ', sentence_list)
        sentence_string = ' '.join(sentence_list)
        generated += sentence_string
        sys.stdout.write(generated)
        sys.stdout.write(" ")
        #시드를 기반으로 텍스트 자동 생성
        for i in range(40):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence_list):
                x[0, t, char_indices[char]] = 1
            #다음에 올 문자를 예측하기
            preds = model.predict(x, verbose = 0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            #출력
            generated = generated  + next_char
            sentence_string = sentence_string[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.write(" ")
            sys.stdout.flush()
        print()









end = time.time()
print(end-start, '초')