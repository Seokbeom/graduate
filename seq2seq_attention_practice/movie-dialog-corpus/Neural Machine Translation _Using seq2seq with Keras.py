


import time
import numpy as np


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


    encoding_input =[]
    decoding_input=[]
    decoding_output=[]
    c=1
    d=1
    while(sentence):
        if len(sentence) <= 200: # 너무 긴 문장은 뺴기...

            sentence_in_list = sentence.lstrip().split(" ")
            encoding_input.append(sentence_in_list)
            decoding_input.append(sentence_in_list[: -1])
            decoding_output.append(sentence_in_list)
            c+=1
        else:
            d+=1

        sentence = data.readline().lstrip()
    print(c, d)
    encoding_input = encoding_input[ : -1] # 차원: 문장수 * 그 문장의 단어수
    decoding_input = decoding_input[1: ]
    decoding_output = decoding_output[1: ]

    data.close()
    return (encoding_input, decoding_input, decoding_output)


def convert_3d_shape_onehotencode(word2one, sentences_list):
    max=49 # 0 으로해놔야함 원래
    for s in sentences_list:
        length = len(s)
        if max < length: #and length < 230: # memory error
            max = length
    print('max ', max)
    X = np.zeros((len(sentences_list), max, len(word2one)), dtype=np.bool)
    for i, sentence in enumerate(sentences_list):
        for j, word in enumerate(sentence):
            #print(j,word)
            try:
                X[i, j, word2one[word]] = 1
            except KeyError as e:
                print('word out of dictionary')
    return X

def sample(preds):
    preds = np.asarray(preds).astype('float64')
    # print(preds)
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, size=1)  # preds의 확률대로 주사위를 1번 던저서 나온 결과를 출력, size 번 반복
    return np.argmax(probas)



def main(data_to_read): # 코드이해 30%
    start = time.time()
    from keras import Model
    from keras.layers import Dense
    from keras.layers import LSTM


    #datatoread = 'test_cleansed.txt'
    data_in_lang = read_data(data_to_read)# 다른 경로에 데이터 몰아넣고 읽게 하자
    word2onehot_dict, one2word_dict = one_hot_dictionary(data_to_read)
    #dic = one_hot_dictionary(data_to_read)
    #word2onehot_dict = dic[0]
    #one2word_dict = dic[1]

    #prepare data which will be used in NN
    encode_input = convert_3d_shape_onehotencode(word2onehot_dict, data_in_lang[0])
    decode_input = convert_3d_shape_onehotencode(word2onehot_dict, data_in_lang[1])
    decode_ouput = convert_3d_shape_onehotencode(word2onehot_dict, data_in_lang[2])

    # Encoder model
    from keras.layers import Input
    total_words = len(word2onehot_dict)
    encoder_input = Input(shape=(None, total_words))
    encoder_LSTM = LSTM(256, return_state=True, return_sequences=False)
    encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input) #<class 'tensorflow.python.framework.ops.Tensor'>
    encoder_states = [encoder_h, encoder_c]


    # Decoder model
    decoder_input = Input(shape=(None, total_words))
    decoder_LSTM = LSTM(256, return_state=True, return_sequences=True)
    decoder_out, _ , _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
    decoder_dense = Dense(total_words, activation='softmax')
    decoder_out = decoder_dense(decoder_out)

    # Train ######  layer.get_weights(): returns the weights of the layer as a list of Numpy arrays.
    model = Model(inputs=[encoder_input, decoder_input], outputs = [decoder_out])
    model.summary()
    model.compile(optimizer='adam', loss= 'categorical_crossentropy') ##
    model.fit(x=[encode_input, decode_input],
              y=decode_ouput,
              batch_size=32, # 64 is too big
              epochs=1,
              validation_split=0.2,
              verbose=2)
    print('fitting finished')
    end = time.time()
    print((end - start) / 60, '분')
    #model.save('seq2seq_no_Attention.model') # package  hdf5


    ### Infernece Encode model
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

        target_seq = np.zeros((1, 1, total_words))
        target_seq[0, 0 , word2onehot_dict['<eos>\n']] = 1

        reply_sentence =''
        stop = False

        while not stop:
            decoder_out, decoder_h, decoder_c = decoder_model_inf.predict( x = [target_seq] + states_val)
            max_val_index = np.argmax(decoder_out[0, -1, :])
            sampled_output = one2word_dict[max_val_index]
            reply_sentence += sampled_output +' '

            if sampled_output == '<eos>\n' or len(reply_sentence)>  49 : # 49 = 제일 긴 문장 수
                stop = True

            target_seq = np.zeros((1, 1, total_words))
            target_seq[0, 0, max_val_index] = 1

            states_val = [decoder_h, decoder_c]


        return reply_sentence

    input_sentence ="a a a a a <eos>\n"
    for turn in range(10):
        try:
            #print("A: " , input_sentence)
            #input_sentence = input("I say : ")
            #if input_sentence == 'q':
            #   break
            input_sentence.strip()
            sen_in_list=[]
            #input_sentence += '<eos>\n'
            print(input_sentence.split(" "))
            temp_sen_in_list = input_sentence.split(' ')
            for word in temp_sen_in_list:
                if word != "":
                    sen_in_list.append(word)

            inp_seq =  [sen_in_list]
            inp_seq = convert_3d_shape_onehotencode(word2onehot_dict, inp_seq)
            result = decode_seq(inp_seq)
            print(turn, result)
            input_sentence = result
        except :
            print('except')


    print('end program')







data_to_read = 'cleansed_movie_dialogue_shrinked.txt'
data_to_read='test_cleansed.txt'
main(data_to_read)
