import re

#extract sen len 11로 하자 걍

def clean(text):
    # Clean the text
    #text = text.replace("\'re", " are ").replace("\'ll", " will ").replace("\'m", " am ").replace("?", " ? ").replace("!", " ! ")
    #text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"i ain't", "i am not ", text)
    text = re.sub(r"you ain't", "you are not ", text)
    text = re.sub(r"they ain't", "they are not ", text)
    text = re.sub(r"we ain't", "we are not ", text)
    text = re.sub(r"he ain't", "he is not ", text)
    text = re.sub(r"she ain't", "she is not ", text)
    text = re.sub(r"it ain't", "it is not ", text)
    text = re.sub(r"this ain't", "this is not ", text)
    text = re.sub(r"those ain't", "those are not ", text)
    text = re.sub(r"these ain't", "these are not ", text)
    text = re.sub(r"ain't", "is not ", text)

    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"is", " s ", text)


    text = re.sub(r"[^A-Za-z0-9!?.]", " ", text)
    text = re.sub("!\.+", "!", text)
    text = re.sub("\?\.+", "?", text)
    text = re.sub("\.!+", ".", text)
    text = re.sub("\.\?+", ".", text)
    text = re.sub("\?!+", "?", text)
    text = re.sub("!\?+", "!", text)
    text = re.sub("\?+", "?", text)
    text = re.sub("!+", "!", text)
    text = re.sub("\?", " ? ", text)
    text = re.sub("!", " ! ", text)
    text = re.sub("\.+", " . ", text)
    text = re.sub("\s+", " ", text)  # 연속되는 공백문자 공백하나로 통일
    text = text.strip()
    #text = wnl.lemmatize(text).lower()

    return text

    ''' 
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
  
    '''
#text = ''.join([c for c in text if c not in punctuation])


sentence ="i was walking the street and saw many cars you ain't we're  can't kill me NAME NAME    it is not mine it's not mine k"
a= clean(sentence)
print(a)
