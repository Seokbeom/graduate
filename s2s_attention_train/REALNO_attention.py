'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi "0000:65:00.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple
'''
import time
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional,Masking, Dropout,TimeDistributed, RepeatVector, Dense
#from attention_decoder import AttentionDecoder
from keras import callbacks


#from machine_learning.parallelizer import Parallelizer  # https://github.com/rmkemker/main
#from keras.utils.training_utils import multi_gpu_model
#-----------------

filename = 'movie_dialogue_15_T_13714.txt'
#filename='test_cleansed_1_T_1.txt'
#filename= 'movie_dialogue_5_T_2715.txt'
filename = 'movie_dialogue_15_T_9752.txt' #q and a + 1024 + dropout한 버전이 성능이 좋은 것 같음...
#filename = 'movie_dialogue_30_T_16549.txt'

#---------------------

senlen = int(filename.split("_")[2])
wordscount = int(filename[:-4].split("_")[4])
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
    return word_to_onehot #onehot_to_word


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


def main(T,Q,A): # 코드이해 30%
    start = time.time()
    global max_step
    data_location = './extracted_data/'
    word2onehot_dict = one_hot_dictionary(data_location + T)
    #word2onehot_dict, one2word_dict = one_hot_dictionary(data_location + T)

    #data_in_lang = read_data_old_version(data_location + T)
    #encode_input = convert_3d_shape_onehotencode(word2onehot_dict, data_in_lang[0])
    #decode_ouput = convert_3d_shape_onehotencode(word2onehot_dict, data_in_lang[1])

    read_data( data_location + T, False)#  to get max step
    encode_input = convert_3d_shape_onehotencode(word2onehot_dict, read_data( data_location + Q))
    decode_ouput = convert_3d_shape_onehotencode(word2onehot_dict, read_data( data_location + A))
    end = time.time()
    print('read and vectorize', (end - start) / 60, '분')

    n_features = len(word2onehot_dict)
    unit = 512
    batchisize = 128
    epoch = 200
    valsplit=0.2
    period = 10

    # define model
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(max_step, n_features))) ##이거할까 #embedin layer를 추가할까
    model.add(Masking(mask_value=0, input_shape=(max_step, n_features)))
    model.add(Bidirectional(LSTM(units=unit, input_shape=(max_step, n_features)), merge_mode='sum'))  #
    model.add(RepeatVector(max_step))
    model.add(LSTM(unit, return_sequences=True))
    model.add(TimeDistributed(Dense(n_features, activation='softmax')))
   #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    '''
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(max_step, n_features))) ##이거할까 #embedin layer를 추가할까
    model.add(Masking(mask_value=0, input_shape=(max_step, n_features)))
    model.add(Bidirectional(LSTM(units= unit, input_shape=(max_step, n_features), return_sequences=True), merge_mode='sum'))#
    model.add(AttentionDecoder(unit, n_features))
    '''
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


    end = time.time()
    print('model construct', (end - start) / 60, '분')

    filepath = './model/' + T[:-4] + '__epoch_{epoch:02d}_loss_{loss:.6f}_valloss_{val_loss:.6f}_acc_{acc:.6f}_QandA_NoAtte.h5'
    callback0 = callbacks.ModelCheckpoint(filepath, monitor='val_loss', period=period)
    callback1 = callbacks.ModelCheckpoint(filepath, monitor='loss', period=period)
    callback2 = callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto')
    callback3 = callbacks.TensorBoard(log_dir='./logs/' + T[:-4] + filepath[-10:-3] + '/', histogram_freq=0,
                                      batch_size=batchisize, write_graph=True, write_grads=False, write_images=False,
                                      embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    model.fit(encode_input, decode_ouput, epochs=epoch, verbose=2, batch_size=batchisize,
              callbacks=[callback0, callback1, callback3], validation_split=valsplit)
    #callback0 = callbacks.ModelCheckpoint(filepath, monitor='val_loss', period=period)
    #callback1 = callbacks.ModelCheckpoint(filepath, monitor='loss', period=period)
    #model.fit(encode_input, decode_ouput, epochs=epoch, verbose=2, batch_size=batchisize, callbacks= [callback0, callback1], validation_split=valsplit)
    #model.save('./model')
    end = time.time()
    print('fitting done', (end - start) / 60, '분')



T = '%s_%s_%d_T_%d.txt'%(t1, t2, senlen, wordscount)
Q = '%s_%s_%d_Q_%d.txt'%(t1, t2, senlen, wordscount)
A = '%s_%s_%d_A_%d.txt'%(t1, t2, senlen, wordscount)
main(T, Q, A)

