import time
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence

#Make sure you have a C compiler before installing gensim, to use optimized (compiled) word2vec training
file_name= 'movie_dialogue_5_T_2715.txt'
size = 100


start = time.time()
data_location = './extracted_data/'
word2vec_location = './word2vec_model/'

data = LineSentence(data_location + file_name)
model = word2vec.Word2Vec(size=size, seed=1234, min_count=1,alpha= 0.025, min_alpha=0.025,  workers=4)
model.build_vocab(data)
#model.train(data,total_examples=model.corpus_count,epochs=model.iter, compute_loss=True)
for epoch in range(10): # in range(10) ? 50 doesnt work // the smaller the better ..... .?
    model.train(data,total_examples=model.corpus_count,epochs=model.iter)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
    print('loss : ', model.get_latest_training_loss())
# To save

model.save(word2vec_location + file_name +'_%s_dim.model'%(str(size)))  #model = word2vec.Word2Vec.load('word2vec_misaeng.model')
print('number of vocab in model: ' , len(model.wv.vocab))
word='you'
print(model.wv.most_similar(positive=[word]))



end = time.time()
print(end-start, '초')