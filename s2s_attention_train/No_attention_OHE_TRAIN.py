
import time
from keras import Model
from keras.layers import Input
from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM , Masking, Dense
from keras import callbacks
import numpy as np



filename = 'movie_dialogue_15_T_9752.txt' #q and a + 1024 + dropout한 버전이 성능이 좋은 것 같음...

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


def main(T, Q, A):
    start = time.time()
    global max_step
    data_location = './extracted_data/'
    word2onehot_dict = one_hot_dictionary(data_location + T)
    end = time.time()
    print('read and vectorize', (end - start) / 60, '분')

  
    read_data(data_location + T, False)  # to get max step
    encode_input = convert_3d_shape_onehotencode(word2onehot_dict, read_data(data_location + Q))
    #decode_input = convert_3d_shape_onehotencode(word2onehot_dict, read_data(data_location + A))
    decode_ouput = convert_3d_shape_onehotencode(word2onehot_dict, read_data(data_location + A))

    n_features = len(word2onehot_dict)
    unit = 512
    batchisize = 128
    epoch = 200
    valsplit = 0.2
    period = 10
    # Encoder model

    n_features = len(word2onehot_dict)
    encoder_input = Input(shape=(None, n_features))

    encoder_LSTM = LSTM( units= unit, return_state=True, return_sequences=False) # binirectional 그냥 빼고
    encoder_outputs, encoder_h, encoder_c = encoder_LSTM (encoder_input)
    encoder_states = [encoder_h, encoder_c]


    # Decoder model
    decoder_input = Input(shape=(None, n_features))
    decoder_LSTM = LSTM(unit, return_state=True, return_sequences=True)
    decoder_out, _ , _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
    decoder_dense = Dense(n_features, activation='softmax')
    decoder_out = decoder_dense(decoder_out)


    model = Model( inputs = [encoder_input, decoder_input], outputs = [decoder_out])
    #model.add(Masking(mask_value=0, input_shape=(max_step, n_features)))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    end = time.time()
    print('model construct', (end - start) / 60, '분')

    filepath = './model/' + T[-4] + '__epoch_{epoch:02d}_loss_{loss:.6f}_valloss_{val_loss:.6f}_acc_{acc:.6f}_QandA_NO_OHE.h5'
    callback0 = callbacks.ModelCheckpoint(filepath, monitor='val_loss', period=period)
    callback1 = callbacks.ModelCheckpoint(filepath, monitor='loss', period=period)
    callback2 = callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto')
    callback3 = callbacks.TensorBoard(log_dir='./logs/' + T[:-4] + filepath[-10:-3] + '/', histogram_freq=0,
                                      batch_size=batchisize, write_graph=True, write_grads=False, write_images=False,
                                      embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    
    model.fit(x=[encode_input, decode_ouput],
              y=decode_ouput,
              batch_size  =batchisize,
              epochs=epoch,
              verbose=2,
              validation_split=valsplit,
              callbacks=[callback0, callback1, callback3],

              )
    print('fitting finished')
    end = time.time()
    print((end - start) / 60, '분')
    print('end program')


    #######################Infernece Encode model

    encoder_model_inf = Model(encoder_input, encoder_states)

    ###Inference Decoder Model
    decoder_state_input_h = Input(shape=(256,))
    decoder_state_input_c = Input(shape=(256,))
    decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

    decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input, initial_state = decoder_input_states)
    decoder_states = [decoder_h , decoder_c]
    decoder_out = decoder_dense(decoder_out)
    decoder_model_inf = Model(inputs=[decoder_input] + decoder_input_states,
                                outputs=[decoder_out] + decoder_states )

    def decode_seq(inp_seq):
        #initial states value is coming from the encoder
        states_val = encoder_model_inf.predict(inp_seq)

        target_seq = np.zeros((1, 1, n_features))
        target_seq[0, 0 , word2onehot_dict['<eos>\n']]= 1

        reply_sentence =''
        stop = False

        while not stop:
            decoder_out, decoder_h, decoder_c = decoder_model_inf.predict( x = [target_seq] + states_val)
            max_val_index = np.argmax(decoder_out[0, -1, :])
            sampled_output = one2word_dict[max_val_index]
            reply_sentence += sampled_output +' '

            if sampled_output == '<eos>\n' or len(reply_sentence)>  22 : # 22 = 제일 긴 문장 수
                stop = True

            target_seq = np.zeros((1, 1, n_features))
            target_seq[0, 0, max_val_index] = 1

            states_val = [decoder_h, decoder_c]

        return reply_sentence

    while 1:
        input_sentence = input("I say : ")
        if input_sentence == 'q':
            break
        input_sentence += ' <eos>\n'
        sen_in_list = input_sentence.split(' ')
        inp_seq =  [sen_in_list]
        inp_seq = convert_3d_shape_onehotencode(word2onehot_dict, inp_seq)
        result = decode_seq(inp_seq)

        print('computer: ', result)

    '''
   


T = '%s_%s_%d_T_%d.txt'%(t1, t2, senlen, wordscount)
Q = '%s_%s_%d_Q_%d.txt'%(t1, t2, senlen, wordscount)
A = '%s_%s_%d_A_%d.txt'%(t1, t2, senlen, wordscount)
main(T,Q,A)

