import time
import one_hot_encodding as OHE
from gensim.models import word2vec
start = time.time()
def one_hot_encoded(length_of_ohe, dec):
    one_hot_encoded_result = []
    for i in range(length_of_ohe):
        if i == dec:
            one_hot_encoded_result.append(1)
        else:
            one_hot_encoded_result.append(0)
    # print(one_hot_encoded_result)
    return one_hot_encoded_result



### make lists for train(test) ###

# moedel data
model = word2vec.Word2Vec.load('word2vec_misaeng.model') # 100 dim word embedding model
length_of_ohe = len(model.wv.vocab)
word_vector = model.wv

# data for one hot encoding
fulltext_list = OHE.fulltext_list
one_hot_encode_dec=OHE.result # OHE.result.index(word)

# list for train = test
input_list=[]
answer_list=[]

for i in range(len(fulltext_list)):
    if i%100 ==0:
        print(i)

    try:
        word = fulltext_list[i]
        input_list.append(word_vector[word])

        #print(word)
        if i < len(fulltext_list) -1 :
            answer_list.append(one_hot_encoded(length_of_ohe, one_hot_encode_dec.index(word) )) # one hot encoded list of next word(word to predict)
        else:
            answer_list.append(one_hot_encoded(length_of_ohe, one_hot_encode_dec.index(word) )) ## how do i set the last value of answer_list...?????
        #print('good: ', fulltext_list[i])
    except KeyError:
        #print( 'error: ' , fulltext_list[i])
        pass

end = time.time()
print(end - start, '초')


### NN build ####

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten #for CNN
from keras.layers import SimpleRNN, LSTM #for RNN
from keras.datasets import mnist
from keras.utils import np_utils

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

####keras 공부가 선행되어야 할듯 example과 documentation으로 먼저 해보자
model4 = Sequential()
model4.add(LSTM(units=100,input_shape=(), return_sequences=True)) #lstm many to many 로 했음.. input layer = look_back, hidden =64,.... so on
model4.add(Dense(units=length_of_ohe, activation='softmax')) # outpt layer???? 결론적으로 나올값은 1개이니까
model4.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae']) #
model4.fit(input_list, answer_list, epochs=5, batch_size=1, verbose=2)
train_predictions = model4.predict('i', batch_size=1) ## 여기에 한 단어가 아니라 문장이 들어가서 그 문장의 단어들을 보고 그 다음 단어나 문장을 예측해야 하는 것 아닌가?
print(train_predictions)




#test_predictions = model4.predict(x_test, batch_size=1)
#trainScore = math.sqrt(mean_squared_error(y_train.reshape(y_train.shape[0]), train_predictions.reshape(train_predictions.shape[0])))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(y_test.reshape(y_test.shape[0]), test_predictions.reshape(test_predictions.shape[0])))
#print('Test Score: %.2f RMSE' % (testScore))


end = time.time()
print(end-start, '초')