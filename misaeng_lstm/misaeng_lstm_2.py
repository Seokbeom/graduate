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


data = open('cleansed_twice.txt', 'r', encoding='utf-8')
import time
a = time.localtime()
timestring = str(a[0]) + str(a[1]) + str(a[2]) + str(a[3]) + str(a[4])

result = open('result_' + timestring + '.txt', 'w', encoding='utf-8')
result.close()

text= data.read()
text=text.replace('\n',' <eos>\n ')
text = text.split(' ')
while '' in text:  # 의미없는 '' 제거
    text.remove('')

print('코퍼스의 길이: ', len(text))



#문자를 하나하나 읽어 들이고 ID 붙이기
#모든 단어들을 중복 없이 딕셔너리에 숫자로 코딩해서 넣음
chars =sorted(list(set(text)))
print('사용되는 문자수 : ', len(chars))
char_indices = dict((c,i) for i, c in enumerate(chars))
indices_char = dict((i ,c ) for i, c in enumerate(chars))


# 후보를 배열에서 꺼내기
def sample(preds):
    preds = np.asarray(preds).astype('float64')
    # print(preds)
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, size=1)  # preds의 확률대로 주사위를 1번 던저서 나온 결과를 출력, size 번 반복
    return np.argmax(probas)


#텍스트를 maxlen 개의 문자로 자르고 다음에 오는 문자 등록하기
for variance in range(3):

    maxlen = 3+variance  # maxlen 개의 단어를 한 문장으로 봄                    ##  #################이걸 잘 조절해야 하는듯######################################
    step =1+variance # step 개 단어 단위로 건너뛰면서 새로운 문장을 정의
    print(maxlen, step)
    sentences = []
    next_chars = []

    '''
    i=0
    temp_text = text
    max_sen = 0
    while i <len(text):
        temp_text= text[i:]
        senlen = temp_text.index("<eos>\n")
        if max_sen < senlen:
            max_sen = senlen
        #print(i, senlen)
        #print((text[i: i + senlen]))
        sentences.append(text[i: i + senlen])
        next_chars.append(text[i + senlen])
        i = i + 1 + senlen
    '''
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])



    print("학습할 구문의 수: ", len(sentences))
    print("텍스트를 ID 백터로 변환합니다 ")
    X= np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
    y=np.zeros((len(sentences), len(chars)), dtype = np.bool)


    for i,sentence in enumerate(sentences): # 명확한 이해 필요 #sentense = list ##########이 훈련 ,검증 셋트를 문장단위로 만들어야 함....
        for t, char in enumerate(sentence): #  char = string


            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    #모델 구축

    print("모델  구축")
    model = Sequential()
    model.add(LSTM(units = 256, input_shape = (None, len(chars) ), activation='relu', return_sequences= True) ) ## max sen = 22 인데 어떻게 되려나
    model.add(Dropout(0.5)) # overfitting 막기
    model.add(LSTM(units=256, activation= 'relu'))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01) # 0.01?
    model.compile(loss= 'categorical_crossentropy', optimizer = optimizer)



    text_length = len(text)
    for iteration in range(1,60):
        print()
        print('----------------------------')
        print('반복 = ', iteration)
        temp = random.randint(0, text_length-30)
        print('temp', temp)
        start_index = text.index('<eos>\n', temp) +1
        end_index = text.index('<eos>\n', start_index + 1) +1
        print(start_index, end_index)
        model.fit(X,y,batch_size=128, epochs=1, verbose=2) # ???이해하기

        #임의의 시작 텍스트 .. 이걸 사용자가 인풋으로 줄 수 있게 해야 할듯

        generated = ''
        sentence = text[start_index : end_index ] # 임의로 만든 시작 문장
        print('시드 : ', sentence)
        #sentence_string = ' '.join(sentence)
        generated += " ".join(sentence)
        sys.stdout.write(generated)
        sys.stdout.write(" ")

        result = open('result_' + timestring + '.txt', 'a', encoding='utf-8')
        result.write(generated)
        result.write(" ")

        #시드를 기반으로 텍스트 자동 생성
        for i in range(70):
            x = np.zeros((1, len(sentence), len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1
            #다음에 올 문자를 예측하기
            preds = model.predict(x, verbose = 0)[0]
            next_index = sample(preds)
            next_char = indices_char[next_index]
            #출력

            sentence = sentence[1:]
            sentence.append(next_char)
            sys.stdout.write(next_char)
            sys.stdout.write(" ")
            result.write(next_char)
            result.write(" ")

            sys.stdout.flush()
        result.write("\n\n")
        result.close()
        print()









end = time.time()
print(end-start, '초')