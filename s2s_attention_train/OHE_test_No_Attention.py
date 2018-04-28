#from attention_decoder import AttentionDecoder
import time
import numpy as np
from keras.models import load_model

from tensorflow import nn
from sklearn.model_selection import train_test_split
import keras.backend as K
def perplexity(y_true, y_pred):
    return K.pow(2.0, K.mean(nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true, name='perplexity')))

#
modelname = 'movie_dialogue_15_T_9752__epoch_90_loss_1.299994_valloss_3.622185_acc_0.248388_QandA_NoAtte.h5'
modelname='movie_dialogue_15_T_9752__epoch_100_loss_1.558343_valloss_3.466010_Perplexity_24.617961_OHE_OHE_NOA_.h5'
INIT_TALK = 'how are you ?'
INIT_TALK = None

t1 = modelname[:-4].split("_")[0]
t2 = modelname[:-4].split("_")[1]
senlen = int(modelname.split("_")[2])
wordscount = int(modelname.split("_")[4])
data_location = './extracted_data/'
data_to_read ='%s_%s_%d_T_%d.txt'%(t1,t2, senlen, wordscount)
#data_to_read = 'movie_dialogue_10_Extracted__Lemmatized.txt'

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
            try:
                X[i, j, word2one[word]] = 1
            except :
                print('error', 'word = \"%s\"'%(word))
                #print(word)
    return X

def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(100, preds, size=1)  # preds의 확률대로 주사위를 1번 던저서 나온 결과를 출력, size 번 반복
    return np.argmax(probas)



def main(data_to_read, modelname, init = None):
    start = time.time()

    data_location = './extracted_data/'
    model_location = './model2/'

    read_data(data_location + data_to_read, False)
    word2onehot_dict, one2word_dict = one_hot_dictionary(data_location + data_to_read)
    n_features = len(word2onehot_dict)
    model = load_model(model_location + modelname, custom_objects={'perplexity': perplexity})

    end = time.time()
    print('model loaded', (end - start) / 60, '분')



    if init:
        loop = 30
        input_sentence = init + ' <eos>\n'
        print(input_sentence)
    else:
        loop = 1000
        input_sentence = input("\nI say : ") + ' <eos>\n'

    generated = open(model_location + modelname + '_dialogue.txt', 'w', encoding='utf-8')
    generated.write(input_sentence)
    generated.close()
    generated = open(model_location + modelname + '_dialogue.txt', 'a', encoding='utf-8')

    for _ in range(loop):
        # 컴퓨터와 대화하는 모드면 인풋을 받음/ 아닐 경우 컴퓨터의 전 대답이 인풋으로 들어감
        if input_sentence == 'q <eos>\n':
            break
        print('input_sentence :', [input_sentence])
        # 인풋 스트링을 백터화
        sen_in_list = input_sentence.split(' ') # 리스트로 변환
        if sen_in_list[-1] == '':
            sen_in_list=sen_in_list[ : -1]
        if sen_in_list[0] == '':
            sen_in_list = sen_in_list[1 : ]

        input_sentence='' # input_sentence 초기화
        inp_seq = convert_3d_shape_onehotencode(word2onehot_dict, [sen_in_list])
        result = model.predict(inp_seq) # input에 대한 softmax output 0 * maxstep * n_feature 차원
        first_eos = True
        
        for stepidx in range(len(result[0])): #range(max_step):
            #highest = sample(result[0][stepidx])
            highest = np.argmax(result[0][stepidx])
            word = one2word_dict[highest]
            result[0][stepidx] = np.zeros((n_features), dtype=np.bool)

            # eos가 나오면 그 다음에 오는 모든 단어는 전부 0으로 /  eos가 아니면 argmax만 = 1
            if word == "<eos>\n":
                if first_eos:
                    input_sentence = input_sentence + " " + word
                    result[0][stepidx][highest] = 1
                    first_eos = False
                    #print("<eos>", end=" ") # 탭으로 당겨도 됨
                    print(word, end=' ')
                    break
                
            else:
                input_sentence = input_sentence + " " + word # + " "
                result[0][stepidx][highest] = 1
                print (word, end=' ')


        if not init:
            generated.write(input_sentence + '\n')
            input_sentence = input("\nI say : ") + ' <eos>\n'

        generated.write(input_sentence)

        # online training
        #model.fit(inp_seq, result, epochs=1, verbose=0, batch_size=1)
        print()

    generated.close()
    end = time.time()
    print('총 : ', (end - start) / 60, '분')





main(data_to_read, modelname, init = INIT_TALK)
