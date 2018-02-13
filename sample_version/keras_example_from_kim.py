 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten  # for CNN
from keras.layers import SimpleRNN, LSTM  # for RNN
from keras.datasets import mnist
from keras.utils import np_utils

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

"""XOR EXAMPLE"""


def xor_example():
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # variables numpy array
    labels = np.array([0, 1, 1, 0]).reshape((4, 1))  # labels
    labels = np_utils.to_categorical(labels, 2)  # one_hot_encodings - label is not continous..  just intger name

    # [1 0] [0 1]
    # 심리학과 컴퓨터공학과 소프트웨어학과
    # 0 1 2
    # [1 0 0] [0 1 0] [0 0 1]


    model = Sequential()  # forgot
    model.add(Dense(units=32, activation='relu', input_dim=2))  # input layer =2, hidden = 32, activation f = relu
    model.add(Dense(units=2, activation='sigmoid'))  # output layer =2 , activation f =sigmoid

    model.compile(loss='binary_crossentropy',  # class 가 2개이니까
                  optimizer='rmsprop',  #
                  metrics=['accuracy'])
    # Training
    model.fit(data, labels, epochs=2000, batch_size=4)  # epoch = iteration, batch_size = mini batch 에서 그 batch 인듯??
    # Testing
    predictions = model.predict(data, batch_size=4)

    # Evaluation
    print(data)
    print(np.argmax(predictions, axis=1).reshape(
        (4, 1)))  # axis = 1 is row(or column) argmax = 최대치 즉 class로 나타내기위함,   reshape = transpose the vector or matrix


"""시간되면"""


def iris_example():
    # with cross_validation
    print(1)


"""MNIST EXAMPLE"""


def mnist_example():  # 0~9 숫자 판별 # cnn
    ((x_train), (y_train)), ((x_test), (y_test)) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])  # data cleansing
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])

    y_train = np_utils.to_categorical(y_train, 10)  # one_hot_encodings # category = 0~9

    model2 = Sequential()
    model2.add(Conv2D(32, (3, 3), padding='same', data_format='channels_first', input_shape=(1, 28, 28)))
    model2.add(Activation('relu'))
    model2.add(Conv2D(32, (3, 3)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.5))

    model2.add(Conv2D(64, (3, 3), padding='same'))
    model2.add(Activation('relu'))
    model2.add(Conv2D(64, (3, 3)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.5))

    model2.add(Flatten())
    model2.add(Dense(512))
    model2.add(Activation('relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(10))
    model2.add(Activation('softmax'))
    model2.compile(loss='categorical_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

    # Training
    model2.fit(x_train, y_train, epochs=2, batch_size=128)
    # Testing
    predictions = model2.predict(x_test, batch_size=1,
                                 verbose=2)  # verbose 0 = no print /  verbose 1 = (default) print every batch, / verbose 2 = print every epoch(iteration)
    hit = 0
    miss = 0
    for idx in range(y_test.shape[0]):
        if (np.argmax(predictions[idx]) == y_test[idx]):
            hit += 1
        else:
            miss += 1
    print(hit, miss)
    print("accuracy : " + str(float(hit / (y_test.shape[0]))))



"""HELLO EXAMPLE"""  # rnn


def hello_example():
    data = np.array([0, 1, 2, 2])  # hell
    data = np_utils.to_categorical(data, 4)  # one - hot encoding
    data = data.reshape((1, 4, 4))

    labels = np.array([1, 2, 2, 3])  # ello
    labels = np_utils.to_categorical(labels, 4)
    labels = labels.reshape((1, 4, 4))


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
    print(np.argmax(labels, axis=1))
    print(np.argmax(predictions, axis=1))


def passenger_example():  # lstm
    # https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line
    df = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python',
                     skipfooter=3)  # padnad to rad csv and draw a graph with matolt /  skipfooter = 데이터 뒤에서 몇개 뺀다
    data_value = df.values.astype('float32')  # numpy 로 바꾸기

    plt.plot(df)
    plt.show()

    scaler = MinMaxScaler(feature_range=(0, 1))  # normalize form 0~1 노멀라이즈 방식중하나 ...
    data_value = scaler.fit_transform(data_value)

    def create_dataset(data, look_back=1):
        x_data, y_data = [], []

        for idx in range(data.shape[0] - look_back):
            a = data[idx:idx + look_back]
            x_data.append(a)
            y_data.append(data[idx + look_back])
        return np.array(x_data).reshape((len(x_data), 1, look_back)), np.array(y_data).reshape((len(y_data), 1, 1))

    train_size = int(data_value.shape[0] * 2 / 3)
    # test_size = int(data_value.shape[0]-train_size)

    look_back = 3  # 3개에 대하여 1개를 예측 ??

    train_data = data_value[:train_size]
    test_data = data_value[train_size - look_back:]

    x_train, y_train = create_dataset(train_data, look_back)
    print(x_train)
    x_test, y_test = create_dataset(test_data, look_back)

    ###  여태까지 데이터 클랜징 지금부터 모델 ##

    model4 = Sequential()
    model4.add(LSTM(units=64, input_dim=look_back, input_length=1,
                    return_sequences=True))  # lstm many to many 로 했음.. input layer = look_back, hidden =64,.... so on
    model4.add(Dense(units=1))  # outpt layer???? 결론적으로 나올값은 1개이니까
    model4.compile(loss='mean_squared_error',
                   optimizer='adam',
                   metrics=['mae'])  #
    model4.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)
    train_predictions = model4.predict(x_train, batch_size=1)
    test_predictions = model4.predict(x_test, batch_size=1)
    trainScore = math.sqrt(
        mean_squared_error(y_train.reshape(y_train.shape[0]), train_predictions.reshape(train_predictions.shape[0])))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(
        mean_squared_error(y_test.reshape(y_test.shape[0]), test_predictions.reshape(test_predictions.shape[0])))
    print('Test Score: %.2f RMSE' % (testScore))

    # show graph
    predictions = np.concatenate((train_predictions, test_predictions), axis=0)
    predictions = predictions.reshape((predictions.shape[0], 1))
    plt.plot(data_value[look_back:], label='label')
    plt.plot(predictions, label='pred')
    plt.axvline(x=train_size, color='black')
    plt.legend(loc='upper left')
    plt.show()

passenger_example()
