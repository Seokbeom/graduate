import seq2seq
from seq2seq.models import AttentionSeq2Seq
import time
import numpy as np

from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM

from keras.layers import Embedding
# generate a sequence of random integers


from seq2seq.models import Seq2Seq
from keras import optimizers
from keras.layers import TimeDistributed
from keras.layers import Softmax




max_step = 0

def one_hot_dictionary(file_name):
    data = open(file_name, 'r', encoding='utf8')
    whole_text= data.read()
    data.close()

    whole_text= whole_text.split(' ')
    while '' in whole_text:  # 의미없는 '' 제거
        whole_text.remove('')

    #모든 단어들을 중복 없이 딕셔너리에 숫자로 코딩해서 넣음 ,
    words =sorted(list(set(whole_text)))
    word_to_onehot = dict((c,i) for i, c in enumerate(words))
    onehot_to_word = dict((i ,c ) for i, c in enumerate(words))

    print('사용되는 단어수(중복 제거) : ', len(words))
    return word_to_onehot, onehot_to_word


def read_data(file_name):
    data = open(file_name, 'r', encoding='utf8')
    sentence = data.readline().lstrip()
    sentence = sentence.replace('\ufeff','')
    print(sentence.split(" "))



    encoding_input =[]
    decoding_input=[]
    decoding_output=[]
    c=1
    d=0
    global max_step
    while(sentence):
        if len(sentence) <= 200: # 너무 긴 문장은 뺴기...

            sentence_in_list = sentence.lstrip().split(" ")
            if len(sentence_in_list) > max_step:
                max_step = len(sentence_in_list)
                print(max_step)
            encoding_input.append(sentence_in_list[ : ])
            decoding_output.append(sentence_in_list[ : ])
            decoding_input.append(sentence_in_list[ : ])
            c+=1
        else:
            d+=1

        sentence = data.readline().lstrip()
    print(c, d)
    encoding_input = encoding_input[ : -1] # 차원: 문장수 * 그 문장의 단어수
    decoding_input = decoding_input[1: ]
    decoding_output = decoding_output[1: ]

    #print(encoding_input)
    #print(decoding_output)
    data.close()
    return (encoding_input, decoding_input, decoding_output)



def convert_3d_shape_onehotencode(word2one, sentences_list):
    global  max_step # 0 으로해놔야함 원래 22
    for s in sentences_list:
        length = len(s)
        if max_step < length:
            max_step = length

    X = np.zeros((len(sentences_list), max_step, len(word2one)), dtype=np.bool)
    for i, sentence in enumerate(sentences_list):  # 명확한 이해 필요 #sentense = list ##########이 훈련 ,검증 셋트를 문장단위로 만들어야 함....
        for j, word in enumerate(sentence):  # char = string
            try:
                X[i, j, word2one[word]] = 1
            except:
                print("error")
    return X






def main(data_to_read): # 코드이해 30%
    start = time.time()
    from keras import Model
    from keras.layers import Dense
    from keras.layers import LSTM



    #datatoread = 'test_cleansed.txt'
    data_in_lang = read_data(data_to_read)# 다른 경로에 데이터 몰아넣고 읽게 하자
    word2onehot_dict, one2word_dict = one_hot_dictionary(data_to_read)
    global max_step
    #prepare data which will be used in NN
    encode_input = convert_3d_shape_onehotencode(word2onehot_dict, data_in_lang[0])
    decode_input = convert_3d_shape_onehotencode(word2onehot_dict, data_in_lang[1])
    decode_ouput = convert_3d_shape_onehotencode(word2onehot_dict, data_in_lang[2])


    # configure problem
    n_features = len(word2onehot_dict)
    unit = 256
    epoch = 100
    batchisize = 64
    optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model = Sequential()
    model.add( AttentionSeq2Seq(input_shape = (max_step, n_features), hidden_dim=unit, output_length=max_step, output_dim=n_features, depth=2, bidirectional=True))
    #model = AttentionSeq2Seq(input_shape = (max_step, n_features), hidden_dim=unit, output_length=max_step, output_dim=n_features, depth=1, bidirectional=False)
    #model.add(TimeDistributed(Dense(n_features)) )
    model.add(TimeDistributed( Dense(units=n_features, input_shape=(max_step, n_features), activation='softmax')))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit( encode_input, decode_ouput, epochs=epoch, verbose=2, batch_size=batchisize, validation_split=0.2)
    print("fitting done")
    end = time.time()
    print(( end-start) / 60 , '분')


    while 1:
        input_sentence = input("I say : ")
        if input_sentence == 'q':
            break
        input_sentence += ' <eos>\n'
        sen_in_list = input_sentence.split(' ')
        inp_seq = [sen_in_list]
        inp_seq = convert_3d_shape_onehotencode(word2onehot_dict, inp_seq)
        result = model.predict(inp_seq)



        for vec in result[0]:
            #print(vec)
            highest = np.argmax(vec)
            word = one2word_dict[highest]
            print(word, end=' ')
            if word == "<eos>\n":
                break


        print()


#data_to_read='cleansed_test2.txt'
#data_to_read='test_cleansed.txt'
data_to_read='cleansed_cleansed_twice.txt'
main(data_to_read)



