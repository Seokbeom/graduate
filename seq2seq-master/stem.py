from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

print(wnl.lemmatize('is'))

import nltk

a= 'I am John.'

sentences = nltk.sent_tokenize(a)
sentences = [nltk.word_tokenize(sent) for sent in sentences]
sentences = [nltk.pos_tag(sent) for sent in sentences]
for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            print(chunk)
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    pass
                    #print(len(chunk))
                    #a=a.replace(chunk[1][0],'personname')
                    #print(chunk[1][0])
print(a)

def ie_preprocess(document):
    document = ' '.join([i for i in document.split()])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    names = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
    return names