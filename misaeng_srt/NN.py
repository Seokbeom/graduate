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


model4 = Sequential()
model4.add(LSTM(units=100,input_dim=100, input_length=1, return_sequences=True)) #lstm many to many 로 했음.. input layer = look_back, hidden =64,.... so on
model4.add(Dense(units=length_of_ohe, activation='softmax')) # outpt layer???? 결론적으로 나올값은 1개이니까
model4.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae']) #
model4.fit(input_list, answer_list, epochs=5, batch_size=1, verbose=2)
train_predictions = model4.predict('i', batch_size=1) ## 여기에 한 단어가 아니라 문장이 들어가서 그 문장의 단어들을 보고 그 다음 단어나 문장을 예측해야 하는 것 아닌가?
print(train_predictions)


'''
error message
C:/Users/USER/Desktop/graduate_project/graduate/misaeng_srt/NN.py:72: UserWarning: The `input_dim` and `input_length` arguments in recurrent layers are deprecated. Use `input_shape` instead.
  model4.add(LSTM(units=100,input_dim=100, input_length=1, return_sequences=True)) #lstm many to many 로 했음.. input layer = look_back, hidden =64,.... so on
C:/Users/USER/Desktop/graduate_project/graduate/misaeng_srt/NN.py:72: UserWarning: Update your `LSTM` call to the Keras 2 API: `LSTM(units=100, return_sequences=True, input_shape=(1, 100))`
  model4.add(LSTM(units=100,input_dim=100, input_length=1, return_sequences=True)) #lstm many to many 로 했음.. input layer = look_back, hidden =64,.... so on
Traceback (most recent call last):
  File "C:/Users/USER/Desktop/graduate_project/graduate/misaeng_srt/NN.py", line 77, in <module>
    model4.fit(input_list, answer_list, epochs=5, batch_size=1, verbose=2)
  File "C:\Users\USER\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\models.py", line 960, in fit
    validation_steps=validation_steps)
  File "C:\Users\USER\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1574, in fit
    batch_size=batch_size)
  File "C:\Users\USER\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1407, in _standardize_user_data
    exception_prefix='input')
  File "C:\Users\USER\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 88, in _standardize_input_data
    '...')
ValueError: Error when checking model input: the list of Numpy arrays that you are passing to your model is not the size the model expected. Expected to see 1 array(s), but instead got the following list of 117197 arrays: [array([ -2.18989468e+00,   5.37290871e-01,   7.64074504e-01,
         4.17127550e-01,   1.35906696e+00,   1.02916503e+00,
        -2.32217208e-01,   3.20134312e-01,  -3.44986701e+00,
        -1.06704...

'''
#test_predictions = model4.predict(x_test, batch_size=1)
#trainScore = math.sqrt(mean_squared_error(y_train.reshape(y_train.shape[0]), train_predictions.reshape(train_predictions.shape[0])))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(y_test.reshape(y_test.shape[0]), test_predictions.reshape(test_predictions.shape[0])))
#print('Test Score: %.2f RMSE' % (testScore))


end = time.time()
print(end-start, '초')