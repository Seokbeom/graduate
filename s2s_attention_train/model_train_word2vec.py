
import time
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Bidirectional
from attention_decoder import AttentionDecoder
from keras import callbacks
from gensim.models import word2vec
#-----------------
filename='movie_dialogue_5_T_2715.txt'


#---------------------

senlen = int(filename[15])
wordscount = int(filename[19:23])
max_step =0
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

    print('사용되는 단어수(중복 제거, <eos> 포함) : ', len(words))
    return word_to_onehot, onehot_to_word


def read_data(file_name, doreturn = True):
    data = open(file_name, 'r', encoding='utf8')
    sentence = data.readline().lstrip()
    sentence = sentence.replace('\ufeff','')

    if doreturn:
        result = []
        while(sentence):
            sentence_in_list = sentence.lstrip().split(" ")
            result.append(sentence_in_list)
            sentence = data.readline().lstrip()
        data.close()
        return result

    else:
        global max_step
        while (sentence):
            sentence_in_list = sentence.lstrip().split(" ")
            if len(sentence_in_list) > max_step:
                max_step = len(sentence_in_list)
                print(max_step)
            sentence = data.readline().lstrip()
        data.close()




def convert_3d_shape_onehotencode(word2one, sentences_list):
    global max_step
    X = np.zeros((len(sentences_list), max_step, len(word2one)), dtype=np.bool)
    for i, sentence in enumerate(sentences_list):  # 명확한 이해 필요 #sentense = list ##########이 훈련 ,검증 셋트를 문장단위로 만들어야 함....
        for j, word in enumerate(sentence):  # char = string
            #print(j,word)
            try:
                X[i, j, word2one[word]] = 1
            except:
                print("error")
    return X


def convert_3d_shape_word2vec(filename, sentences_list):
    model = word2vec.Word2Vec.load('./word2vec_model/' + filename + '_100_dim.model')
    word_vec = model.wv
    del model

    global max_step
    X = np.zeros((len(sentences_list), max_step, 100), dtype=np.bool)
    for i, sentence in enumerate(sentences_list):  # 명확한 이해 필요 #sentense = list ##########이 훈련 ,검증 셋트를 문장단위로 만들어야 함....
        for j, word in enumerate(sentence):  # char = string
            if word == '<eos>\n':
                word = '<eos>'
            try:
                X[i, j] = word_vec[word]
            except KeyError as E:
                print(E)
    return X

def main(T,Q,A): # 코드이해 30%
    start = time.time()
    global max_step
    data_location = './extracted_data/'
    read_data( data_location + T,False)#  to get max step
    word2onehot_dict, one2word_dict = one_hot_dictionary( data_location + T)
    encode_input = convert_3d_shape_word2vec(T, read_data( data_location + Q))
    decode_ouput = convert_3d_shape_onehotencode(word2onehot_dict, read_data( data_location + A))

    end = time.time()
    print('read and vectorize', (end - start) / 60, '분')

    embedded_dim = 100
    n_features = len(word2onehot_dict)
    unit = 32
    batchisize = 32
    epoch = 50
    period = 10

    model = Sequential()
    model.add(Dropout(0.2, input_shape=(max_step, embedded_dim)))
    model.add(Bidirectional(LSTM(unit, input_shape=(max_step, embedded_dim), return_sequences=True), merge_mode='sum'))#
    model.add(AttentionDecoder(unit, n_features))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


    filepath ='./model/'+  T[:-4] + '__epoch_{epoch:02d}_loss_{loss:.3f}_valloss_{val_loss:.3f}_W2V.h5'
    callback = callbacks.ModelCheckpoint(filepath, monitor='val_loss', period=period)
    callback2 = callbacks.ModelCheckpoint(filepath, monitor='loss', period=period)

    model.fit(encode_input, decode_ouput, epochs=epoch, verbose=2, batch_size=batchisize, callbacks= [callback, callback2], validation_split=0.2)

    end = time.time()
    print( 'model saved ',(end - start) / 60, '분')








T = 'movie_dialogue_%d_T_%d.txt'%(senlen, wordscount)
Q = 'movie_dialogue_%d_Q_%d.txt'%(senlen, wordscount)
A = 'movie_dialogue_%d_A_%d.txt'%(senlen, wordscount)
main(T, Q, A)
