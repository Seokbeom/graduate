'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi "0000:65:00.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1" for multiple
'''
import time
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Bidirectional, Masking, Lambda #Dense, activations
from attention_decoder_tanh  import AttentionDecoder
from keras import callbacks
from gensim.models import word2vec
from keras import backend as K
from keras.optimizers import Adam


#-----------------
#filename='movie_dialogue_5_T_2715.txt' # QandA version, diffent dimension sizes , etc....
filename = 'movie_dialogue_15_T_9752.txt'
#---------------------
print('word2vec train')
t1 = filename[:-4].split("_")[0]
t2 = filename[:-4].split("_")[1]
senlen = int(filename[:-4].split("_")[2]) #int(filename[15:17])
wordscount = int(filename[:-4].split("_")[-1])  #int(filename[20:24])

max_step =0

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




def convert_3d_shape_word2vec(filename, sentences_list):
    model = word2vec.Word2Vec.load('./word2vec_model/' + filename + '_100_dim.model')
    model.delete_temporary_training_data(replace_word_vectors_with_normalized=True)  # normalize
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


    read_data(data_location +T, False)
    encode_input = convert_3d_shape_word2vec(T, read_data( data_location + Q))
    decode_ouput = convert_3d_shape_word2vec(T, read_data( data_location + A))
    end = time.time()
    print('read and vectorize', (end - start) / 60, '분')



    embedded_dim = 100
    unit = 512
    batchisize = 128
    epoch = 500
    valsplit = 0.2
    period = 10

    model = Sequential()
    model.add(Dropout(0.3, input_shape=(max_step, embedded_dim))) #0.5?
    model.add(Masking(mask_value=0., input_shape=(max_step, embedded_dim)))
    model.add(Bidirectional(LSTM(unit, input_shape=(max_step, embedded_dim), return_sequences=True), merge_mode='sum'))#
    model.add(AttentionDecoder(unit, embedded_dim))
    model.add(Lambda(lambda x : K.l2_normalize(x) ))
    optimizer = Adam(lr=0.006, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)#default = 0.001
    #model.add(Lambda(lambda x: x / ( K.l2_normalize(x) * K.l2_normalize(x) )))
    #model.add(Dense(units=unit, input_shape=(max_step, embedded_dim), activation='tanh'))# model.add(TimeDistributed( Dense(units=n_features, input_shape=(max_step, n_features), activation='softmax')))
    model.summary()
    model.compile(loss='cosine_proximity', optimizer=optimizer, metrics=['acc'])


    filepath ='./model/'+  T[:-4] + '__epoch_{epoch:02d}_loss_{loss:.6f}_valloss_{val_loss:.6f}_acc_{acc:.6f}_cccc.h5'
    callback0 = callbacks.ModelCheckpoint(filepath, monitor='val_loss', period=period)
    callback1 = callbacks.ModelCheckpoint(filepath, monitor='loss', period=period)
    callback2 = callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto')
    callback3 = callbacks.TensorBoard(log_dir='./logs/'+T[ :-4] + filepath[-8:-3]+'/', histogram_freq=0, batch_size=batchisize, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    model.fit(encode_input, decode_ouput, epochs=epoch, verbose=2, batch_size=batchisize,
              callbacks=[callback0, callback1, callback3], validation_split=valsplit)




    end = time.time()
    print( 'model saved ',(end - start) / 60, '분')








T = '%s_%s_%d_T_%d.txt'%(t1, t2, senlen, wordscount)
Q = '%s_%s_%d_Q_%d.txt'%(t1, t2,senlen, wordscount)
A = '%s_%s_%d_A_%d.txt'%(t1, t2, senlen, wordscount)
main(T, Q, A)
