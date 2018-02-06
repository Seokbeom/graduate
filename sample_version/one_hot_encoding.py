## -----  to make full_text_list  ----- ##
data = open('cleansed.txt', 'r', encoding='utf-8')
fulltext= data.read()
fulltext=fulltext.replace('\n',' ') # ' \n ' ??
fulltext_list = fulltext.split(' ') ### variable (later used in another files)

#with open("full_text_list.pkl", "wb") as fp:
    #pickle.dump(fulltext_list, fp)






## -----  to make original words list and original words in decimal list ----- ##
from gensim.models import word2vec
model = word2vec.Word2Vec.load('word2vec_vectorized.model') # 100 dim word embedding model
word_vector = model.wv.vocab
original_words=[]
decimal_words=[]
num = 0
for i in word_vector:
    original_words.append(i)
    decimal_words.append(num)
    num += 1

print('words_in_english: ' , original_words)
print('words_in_decimal: ' , decimal_words)

import pickle
with open("words_in_english.pkl", "wb") as fp:
    pickle.dump(original_words, fp)
with open("words_in_decimal.pkl", "wb") as fp:
    pickle.dump(decimal_words, fp)



## -----  one_hot_encoding the full text with the encoded words above  && make input data list(wordsnumber*100 dim) ----- ##

word_num = 0
fulltext_dec=[]
input_for_nn=[]
for i in fulltext_list:
    try:
        input_for_nn.append(word_vector[i])
        fulltext_dec.append(original_words.index(i))
    except :
        pass
words_num = len(fulltext_dec)

from keras.utils import np_utils
fulltext_ohe= np_utils.to_categorical(fulltext_dec, words_num) ## one hot encoding을 이걸로 하는구나...
fulltext_ohe = fulltext_ohe.reshape((1, words_num, words_num))

print('full text in OHE: \n', fulltext_ohe)
print('input_for_nn: ', input_for_nn)

with open("fulltext_in_OHE.pkl", "wb") as fp:
    pickle.dump(fulltext_ohe, fp)
with open("input_for_nn.pkl", "wb") as fp:
    pickle.dump(input_for_nn, fp)









