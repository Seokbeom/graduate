# wikibooks p 255
import time
start = time.time()

from keras.datasets import mnist # 숫자 손글씨 데이터
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

#MNIST 데이터 읽기 Modified National Institute of Standards and Technology database)
(X_train, y_train), (X_test, y_test) = mnist.load_data() # 한번만 하면 됨

#이미지 데이터가 어떻게 생겼는지 확인하기
print(type(X_train)) # class numpy.ndarray
print(X_train.shape) # 6000개 데이터 / 28pix / 28pix
print(X_test.shape) # 1000개 데이터 / 28pix / 28pix

#이미지 데이터를 다루기 쉬운 모양으로 변환
X_train = X_train.reshape(X_train.shape[0], 1 ,X_train.shape[1], X_train.shape[2]) # 6000 / 1 / 28 / 28  차원의 행렬로 변환?
X_test = X_test.reshape(X_test.shape[0] , 1 ,X_test.shape[1], X_test.shape[2]) # 1000 / 1 / 28 / 28

#정답데이터를 one hot encoding 하기 # one_hot_encoding -> ex) 0 = 1000000000 , 1 = 0100000000, 2 = 0010000000 ....
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)
print(y_train)

#모델 구조 정의하기 (https://keras.io/layers/convolutional)
#--- Convolutional part ( 2d 데이터를 filtering 및 pooling ) ---#
from keras.layers import Conv2D, MaxPooling2D, Flatten   # for CNN
model = Sequential() # 데이터의 순서가 중요한 sequentail
model.add(Conv2D(32, (3,3), padding ='same', data_format = 'channels_first', input_shape=(1,28,28) )) # 3*3필터 32개(랜덤?)
                                                                                                            # #padding은 input과 output이 같게(https://www.coursera.org/learn/convolutional-neural-networks/lecture/o7CWi/padding)
                                                                                                            #  #data_format = 첫 레이어라는건가?
                                                                                                            # input_shpae = 데이터 1개 28*28차원

model.add(Activation('relu')) #해당 layer의 avtivation function
model.add(Conv2D(32,(3,3))) # filter 한번 더 하기
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2))) # 2*2 mask로 max pooling( 4 pixel 값 중 최대치만 대표로 뽑음 )
model.add(Dropout(0.5)) # overfitting 막기

#--- 여기서부터 Dense part (사실상 학습이 일어나는 곳 ??) ---#
model.add(Flatten()) # 2차원 데이터를 1차원으로 쭉 늘림
model.add(Dense(512)) #fully connected layer
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', # loss 정의
                optimizer = 'rmsprop', #optimizer 의 한 종류(gardient decent method = sgd)
                metrics=['accuracy']) # 정답률을 출력

#Training
model.fit(X_train, y_train, epochs=2, batch_size=128)
#Testing
import numpy as np
predictions = model.predict(X_test, batch_size =1, verbose=2)

hit=0
miss=0
for idx in range(y_test.shape[0]):
    if (np.argmax(predictions[idx]) == y_test[idx]):
        hit +=1
    else:
        miss +=1

print(hit, miss)
print("accuarcy : " , str(float(hit / (y_test.shape[0]))))






end = time.time()
print(end-start, '초')