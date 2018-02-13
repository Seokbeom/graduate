
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten  # for CNN
from keras.layers import SimpleRNN, LSTM  # for RNN
from keras.datasets import mnist
from keras.utils import np_utils


import numpy as np



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
