import numpy as np
a =np.arange(15).reshape(3,5) ## 3*5 matrix
print(a)

b=np.array([6,7,8])
print(b)

c= np.empty((2,3))
print(c)
print('-----------------------')
data = np.array([0, 1, 2, 2])  # hell
from keras.utils import np_utils
data = np_utils.to_categorical(data, 4) ## one hot encoding을 이걸로 하는구나...
data = data.reshape((1, 4, 4))
print(data)
print(type(data))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten #for CNN
from keras.layers import SimpleRNN, LSTM #for RNN
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
model3.fit(data, labels, epochs=10, batch_size=1, verbose=0)


predictions = model3.predict(data, batch_size=1)
predictions = predictions.reshape((4, 4))
labels2 = labels.reshape((4, 4))
print(predictions)
print('정답 : ', np.argmax(labels2, axis=1))
print('예측 값 : ' , np.argmax(predictions, axis=1))



model3.fit(data, labels, epochs=100, batch_size=1,  verbose=0)
predictions = model3.predict(data, batch_size=1)
predictions = predictions.reshape((4, 4))
labels2 = labels.reshape((4, 4))
print(predictions)
print('정답 : ', np.argmax(labels2, axis=1))
print('예측 값 : ' , np.argmax(predictions, axis=1))



