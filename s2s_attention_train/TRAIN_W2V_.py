#import tensorflow as tf
#with tf.device('/gpu:0'):
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi #"0000:17:00.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1" for multiple
import time
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Bidirectional, Masking
from attention_decoder import AttentionDecoder
from keras import callbacks
from gensim.models import word2vec

from tensorflow import nn
from sklearn.model_selection import train_test_split
import keras.backend as K
def perplexity(y_true, y_pred):
    return K.pow(2.0, K.mean(nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, name='perplexity')))

#

#-----------------
#filename='movie_dialogue_10_T_9188.txt' # QandA version, diffent dimension sizes , etc....
filename = 'movie_dialogue_15_T_9752.txt'
#---------------------
print('word2vec train')
senlen = int(filename[:-4].split("_")[2]) #int(filename[15:17])
wordscount = int(filename[:-4].split("_")[-1])  #int(filename[20:24])
t1 = filename[:-4].split("_")[0]
t2 = filename[:-4].split("_")[1]
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
    #onehot_to_word = dict((i ,c ) for i, c in enumerate(words))

    print('사용되는 단어수(중복 제거, <eos> 포함) : ', len(words))
    return word_to_onehot #, onehot_to_word

def read_data_old_version(file_name):
    data = open(file_name, 'r', encoding='utf8')
    sentence = data.readline().lstrip()
    sentence = sentence.replace('\ufeff','')
    #print(sentence.split(" "))



    encoding_input =[]
    decoding_output=[]
    global max_step
    while(sentence):


        sentence_in_list = sentence.lstrip().split(" ")
        if len(sentence_in_list) > max_step:
            max_step = len(sentence_in_list)
            print(max_step)
        encoding_input.append(sentence_in_list)
        decoding_output.append(sentence_in_list)
        sentence = data.readline().lstrip()

    encoding_input = encoding_input[ : -1] # 차원: 문장수 * 그 문장의 단어수
    decoding_output = decoding_output[1: ]

    data.close()
    return (encoding_input, decoding_output)

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
    X = np.zeros((len(sentences_list), max_step, 100), dtype=np.float)
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
    word2onehot_dict = one_hot_dictionary(data_location + T)
    # word2onehot_dict, one2word_dict = one_hot_dictionary( data_location + T)


    #data_in_lang = read_data_old_version( data_location + T)
    #encode_input = convert_3d_shape_word2vec(T, data_in_lang[0])
    #decode_ouput = convert_3d_shape_onehotencode(word2onehot_dict, data_in_lang[1])

    read_data(data_location +T, False)
    encode_input = convert_3d_shape_word2vec(T, read_data( data_location + Q))
    decode_ouput = convert_3d_shape_onehotencode(word2onehot_dict, read_data( data_location + A))
    end = time.time()
    print('read and vectorize', (end - start) / 60, '분')

    from parameters import parameters
    n_features = len(word2onehot_dict)
    embedded_dim = 100
    unit = 512
    batchisize = parameters.batchisize
    epoch = 200
    valsplit = 0.1
    period = 10

    model = Sequential()
    model.add(Dropout(0.33, input_shape=(max_step, embedded_dim)))
    model.add(Masking(mask_value=0., input_shape=(max_step, embedded_dim)))
    model.add(Bidirectional(LSTM(unit, input_shape=(max_step, embedded_dim), return_sequences=True), merge_mode='concat'))#
    model.add(AttentionDecoder(unit, n_features))
    model.summary()
    #
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[perplexity])
    end = time.time()
    print('model construct', (end - start) / 60, '분')
    filepath = './model2/' + T[
                            :-4] + '__epoch_{epoch:02d}_loss_{loss:.6f}_valloss_{val_loss:.6f}_Perplexity_{perplexity:.6f}_W2V_ORI_ATT_.h5'  # name
    callback0 = callbacks.ModelCheckpoint(filepath, monitor='val_loss', period=period)
    callback1 = callbacks.ModelCheckpoint(filepath, monitor='loss', period=period)
    callback2 = callbacks.ModelCheckpoint(filepath, monitor='perplexity', period=period)
    # callback2 = callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto')
    callback3 = callbacks.TensorBoard(log_dir='./logs/' + T[:-4] + filepath[-16:-3] + '/', histogram_freq=0,
                                      batch_size=batchisize, write_graph=False, write_grads=False, write_images=False,
                                      embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    X_train, X_test, y_train, y_test = train_test_split(encode_input, decode_ouput, test_size=0.2, random_state=7)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, verbose=2, batch_size=batchisize,
              callbacks=[callback0, callback1, callback2, callback3])
    #

    end = time.time()
    print( 'model saved ',(end - start) / 60, '분')








T = 'movie_dialogue_%d_T_%d.txt'%(senlen, wordscount)
Q = 'movie_dialogue_%d_Q_%d.txt'%(senlen, wordscount)
A = 'movie_dialogue_%d_A_%d.txt'%(senlen, wordscount)
main(T, Q, A)
