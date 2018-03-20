#https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/

from random import randint
from numpy import array
from numpy import argmax


#generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [ randint(0, n_unique-1) for _ in range(length)]

#One hot encode
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence: # 이 예시의 sequence 에 있는 값은 모드 integer..
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

#decode one hot
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


sequence = generate_sequence(5, 50)
print(sequence)
encoded = one_hot_encode(sequence, 50)
print(encoded)
decoded = one_hot_decode(encoded)
print(decoded)