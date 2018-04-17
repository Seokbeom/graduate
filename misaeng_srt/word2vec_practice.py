
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence


model = word2vec.Word2Vec.load('word2vec_misaeng.model')

word_vec = model.wv
del model

print(type(word_vec)) # <class 'gensim.models.keyedvectors.Word2VecKeyedVectors'>
print(type(word_vec['you']))  # <class 'numpy.ndarray'>