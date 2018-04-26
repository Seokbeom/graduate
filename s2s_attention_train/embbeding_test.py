import time
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence


modelname ='movie_dialogue_10_T_9188.txt_100_dim.model'
start = time.time()
word2vec_location = './word2vec_model/'

model = word2vec.Word2Vec.load(word2vec_location + modelname)
for word in ['i', 'run', 'hi', '<eos>', 'happy']:
    print(model.wv.most_similar(word))
#print('loss : ', model.get_latest_training_loss())



# To save

#model.save(word2vec_location + file_name +'_%s__dim.model'%(str(size)))  #model = word2vec.Word2Vec.load('word2vec_misaeng.model')
#print('number of vocab in model: ' , len(model.wv.vocab))



end = time.time()
print(end-start, '초')