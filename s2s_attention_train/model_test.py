from keras.models import Sequential


from attention_decoder import AttentionDecoder
import time
import numpy as np


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

    print('사용되는 단어수(중복 제거) : ', len(words))
    return word_to_onehot, onehot_to_word


def read_data(file_name):
    data = open(file_name, 'r', encoding='utf8')
    sentence = data.readline().lstrip()
    sentence = sentence.replace('\ufeff','')
    #print(sentence.split(" "))



    encoding_input =[]
    decoding_input=[]
    decoding_output=[]
    global max_step
    while(sentence):


        sentence_in_list = sentence.lstrip().split(" ")
        if len(sentence_in_list) > max_step:
            max_step = len(sentence_in_list)
            print(max_step)
        encoding_input.append(sentence_in_list[ : ])
        decoding_output.append(sentence_in_list[ : ])
        decoding_input.append(sentence_in_list[: ])
        sentence = data.readline().lstrip()

    encoding_input = encoding_input[ : -1] # 차원: 문장수 * 그 문장의 단어수
    decoding_input = decoding_input[1: ]
    decoding_output = decoding_output[1: ]

    data.close()
    return (encoding_input, decoding_input, decoding_output)





def convert_3d_shape_onehotencode(word2one, sentences_list):
    global  max_step
    X = np.zeros((len(sentences_list), max_step, len(word2one)), dtype=np.bool)
    for i, sentence in enumerate(sentences_list):  # 명확한 이해 필요 #sentense = list ##########이 훈련 ,검증 셋트를 문장단위로 만들어야 함....
        for j, word in enumerate(sentence):  # char = string
            try:
                X[i, j, word2one[word]] = 1
            except:
                print("error")
    return X

def sample(preds):
    preds = np.asarray(preds).astype('float64')
    # print(preds)
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, size=1)  # preds의 확률대로 주사위를 1번 던저서 나온 결과를 출력, size 번 반복
    return np.argmax(probas)


from keras.layers import Input,TimeDistributed
def main(data_to_read): # 코드이해 30%
    start = time.time()

    #datatoread = 'test_cleansed.txt'
    data_in_lang = read_data(data_to_read)# 다른 경로에 데이터 몰아넣고 읽게 하자
    word2onehot_dict, one2word_dict = one_hot_dictionary(data_to_read)
    encode_input = convert_3d_shape_onehotencode(word2onehot_dict, data_in_lang[0])
    decode_ouput = convert_3d_shape_onehotencode(word2onehot_dict, data_in_lang[2])


    # configure problem
    n_features = len(word2onehot_dict)
    unit = 256
    epoch = 100
    batchisize = 64


    global max_step
    print(n_features)
    from keras.models import load_model
    model = load_model('movie_dialogue_10_Extracted__Lemmatizedone_hot.h5', custom_objects={'AttentionDecoder': AttentionDecoder})
    end = time.time()
    print( (end - start) / 60, '분')


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



    print('end program')







#data_to_read = 'cleansed_movie_dialogue_shrinked.txt'
#data_to_read='cleansed_test2.txt'
#data_to_read='test_cleansed.txt'
#data_to_read='cleansed_cleansed_twice.txt'
#data_to_read='movie_dialogue_10_shrinked.txt'
data_to_read='movie_dialogue_10_Extracted__Lemmatized.txt'
main(data_to_read)
