import time
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import numpy as np
#Make sure you have a C compiler before installing gensim, to use optimized (compiled) word2vec training
file_name= 'movie_dialogue_5_T_2715.txt'
#file_name='movie_dialogue_15_T_9752.txt'
size = 100


start = time.time()
data_location = './extracted_data/'
word2vec_location = './word2vec_model/'

data = LineSentence(data_location + file_name)
model = word2vec.Word2Vec(size=size, seed=1234, min_count=1,alpha= 0.025, min_alpha=0.025,  workers=8) ## Try alpha=0.05 and cbow_mean=1  https://stackoverflow.com/questions/34249586/the-accuracy-test-of-word2vec-in-gensim

model.build_vocab(data)
#model.train(data,total_examples=model.corpus_count,epochs=model.iter, compute_loss=True)
for epoch in range(10): # in range(10) ? 50 doesnt work // the smaller the better ..... .?
    model.train(data,total_examples=model.corpus_count,epochs=model.iter)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay


    #print('loss : ', model.get_latest_training_loss())
word = 'i'
print(model.wv.most_similar(word))
#print(model['i'])
model.delete_temporary_training_data(replace_word_vectors_with_normalized=True) # normalize
'''
a = model.wv.vocab
#print(model['i'])
#for i in a:
#    print(model[i])
aa= [ 0.1 * i for i in range(100)]
veotor =np.array(aa)
#word = np.array((1,2,3,),dtype=float)
b = model.wv.get_vector(word)
r = 0
for i in b:
    r = r + i* i
print(r)
print(model.wv.most_similar_cosmul(word))
print(model.wv.most_similar( (b, b),topn=1, )) ## vector b와 제일 근접한 단어 출력(cosine sim)
'''
# To save


model.save(word2vec_location + file_name +'_%s_dim.model'%(str(size)))  #model = word2vec.Word2Vec.load('word2vec_misaeng.model')
print('number of vocab in model: ' , len(model.wv.vocab))



end = time.time()
print(end-start, '초')