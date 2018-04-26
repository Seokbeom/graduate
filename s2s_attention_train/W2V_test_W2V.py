from attention_decoder_tanh import AttentionDecoder
import time
import numpy as np
from keras.models import load_model
from gensim.models import word2vec # Try alpha=0.05 and cbow_mean=1  https://stackoverflow.com/questions/34249586/the-accuracy-test-of-word2vec-in-gensim

#############################
modelname = 'movie_dialogue_15_T_9752__epoch_490_loss_-0.323665_valloss_-0.111473_acc_0.170399_WWWW.h5'
modelname='movie_dialogue_15_T_9752__epoch_100_loss_-0.127914_valloss_-0.128073_acc_0.027951_cccc.h5'
modelname= 'movie_dialogue_15_T_9752__epoch_400_loss_-0.276557_valloss_-0.121348_acc_0.126529_nnnn.h5'

INIT_TALK = 'how are you ?'
INIT_TALK = None

#########################

t1 = modelname[:-4].split("_")[0]
t2 = modelname[:-4].split("_")[1]
senlen = int(modelname.split("_")[2])
wordscount = int(modelname.split("_")[4])
data_location = './extracted_data/'
data_to_read ='%s_%s_%d_T_%d.txt'%(t1,t2, senlen, wordscount)
max_step = 0



def read_data(file_name, doreturn=True):
    data = open(file_name, 'r', encoding='utf8')
    sentence = data.readline().lstrip()
    sentence = sentence.replace('\ufeff', '')

    if doreturn:
        result = []
        while (sentence):
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




def main(data_to_read, modelname, init=None):
    start = time.time()
    model2 = word2vec.Word2Vec.load('./word2vec_model/' + data_to_read + '_100_dim.model')
    model2.delete_temporary_training_data(replace_word_vectors_with_normalized=True)  # normalize
    data_location = './extracted_data/'
    model_location = './model/'

    read_data(data_location + data_to_read, False)  # to get max step
    model = load_model(model_location + modelname, custom_objects={'AttentionDecoder': AttentionDecoder})

    end = time.time()
    print('model loaded', (end - start) / 60, '분')

    if init:
        loop = 30
        input_sentence = init + ' <eos>'
    else:
        loop = 1000
        input_sentence = input("I say : ") + ' <eos>'

    generated = open(model_location + modelname + '_dialogue.txt', 'w', encoding='utf-8')
    generated.write(input_sentence)
    generated.close()
    generated = open(model_location + modelname + '_dialogue.txt', 'a', encoding='utf-8')

    for _ in range(loop):
        # 컴퓨터와 대화하는 모드면 인풋을 받음/ 아닐 경우 컴퓨터의 전 대답이 인풋으로 들어감
        if input_sentence == 'q <eos>':
            break

        # 인풋 스트링을 백터화
        sen_in_list = input_sentence.split(' ')  # 리스트로 변환
        if sen_in_list[-1] == '':
            sen_in_list = sen_in_list[: -1]
        if sen_in_list[0] == '':
            sen_in_list = sen_in_list[1:]

        input_sentence = ''  # input_sentence 초기화
        inp_seq = convert_3d_shape_word2vec(data_to_read, [sen_in_list])
        result = model.predict(inp_seq)  # input에 대한 softmax output 0 * maxstep * n_feature 차원
        #print(result[[0][0]])
        first_eos = True

        for stepidx in range(max_step):
            vector = result[0][stepidx]
            word = model2.wv.most_similar((vector, vector), topn=1)[0][0]  ## vector b와 제일 근접한 단어 출력(cosine sim)
            #print('word: ', word[0][0])
            #word = one2word_dict[highest]
            result[0][stepidx] = np.zeros((100), dtype=np.float)

            # eos가 나오면 그 다음에 오는 모든 단어는 전부 0으로 /  eos가 아니면 argmax만 = 1
            if word == "<eos>":
                if first_eos:
                    input_sentence = input_sentence + " " + word
                    #result[0][stepidx][highest] = 1
                    first_eos = False
                    print(word, end=" ")
                    break

            else:
                input_sentence = input_sentence + " " + word

                print(word, end=' ')

        if not init:
            generated.write(input_sentence + '\n')
            input_sentence = input("\nI say : ") + ' <eos>'

        generated.write(input_sentence+'\n')

        # online training
        #model.fit(inp_seq, result, epochs=1, verbose=0, batch_size=1)
        print()

    generated.close()
    end = time.time()
    print('총 : ', (end - start) / 60, '분')


main(data_to_read, modelname, init=INIT_TALK)
