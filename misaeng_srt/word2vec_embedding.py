import time
import one_hot_encodding as OHE

start = time.time()


from gensim.models import word2vec
from gensim.models.word2vec import LineSentence


 #word2vec model
data = LineSentence('misaeng_cleansed.txt')
model = word2vec.Word2Vec(size=100, alpha=0.025, min_alpha=0.025, seed=1234, workers=4, min_count=1)
#size = 100 is better than 300 so far i guess ### min_count=1 해도 4단어 누락..?
model.build_vocab(data)

for epoch in range(10): # in range(10) ? 50 doesnt work // the smaller the better ..... .?
    model.train(data,total_examples=model.corpus_count,epochs=model.iter)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
# To save
model.save('word2vec_misaeng.model')


model = word2vec.Word2Vec.load('word2vec_misaeng.model')

print('number of vocab in model: ' , len(model.wv.vocab))


word='you'
print(model.wv.most_similar(positive=[word]))
print((model.wv[word]))
print(len(model.wv[word]))
#print('one-hot-encoding : ', OHE.result.index(word)) # 이걸  원핫인코딩하면 됨

end = time.time()
print(end-start, '초')