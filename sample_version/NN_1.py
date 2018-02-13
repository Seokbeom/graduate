import time

#from gensim.models import word2vec

start = time.time()
import pickle
input_data=[]
output_data=[]


with open("input_for_nn.pkl","rb") as fp: # input이 단어 하나 당 100차원의 백터인데 n개의 전 단어로 다음 1개의 단어를 예측하려면 600개의 인풋을 주어야 하나????/
    intput_data = pickle.load(input_data)

with open("fulltext_in_OHE.pkl","rb") as fp:
    output_data = pickle.load(fp)

print(output_data)
print(type(output_data))
### NN build ####
'''
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


model3 = Sequential()
# RNN option에 대해 설명 필요할듯
model3.add(SimpleRNN(units=32, input_dim=4, input_length=4,
                     return_sequences=True))  # units= hidden layer, input_dim= input node,  input_length 인풋의 연속적인 길이 (문맥 파악하는 길이), return_sequences= 다대다인지 다대일인지
model3.add(Dense(units=4, activation='softmax'))  # output layer
model3.compile(loss='categorical_crossentropy',  # class = H,E,L,O
               optimizer='sgd',
               metrics=['accuracy'])
model3.fit(data, labels, epochs=700, batch_size=1)
predictions = model3.predict(data, batch_size=1)
predictions = predictions.reshape((4, 4))
labels = labels.reshape((4, 4))
print(predictions)
print('정답 : ', np.argmax(labels, axis=1))
print('예측 값 : ' , np.argmax(predictions, axis=1))

'''


end = time.time()
print(end-start, '초')