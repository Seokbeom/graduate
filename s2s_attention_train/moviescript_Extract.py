import re
sen_length = 5


def get_id2line():
    lines = open('./data/movie_lines.tsv',encoding='utf8').read()[1:].split('\n')

    #rint(lines[0])
    id2line = {}
    for line in lines:
        _line = line.split('\t')
        #print(_line)
        if len(_line) >= 5: # ==
            id2line[_line[0]] = _line[4]
    #print(id2line)
    return id2line


def get_conversations():
    #a = open('check.txt','w',encoding='utf8')
    #a.close()
    conv_lines = open('./data/movie_conversations.tsv', encoding='utf8').read().split('\n')

    convs = []
    for line in conv_lines[:-1]:
        #print(line)
        #a = open('check.txt', 'a', encoding='utf8')
        #print(line.split('\t')[-1][1:-1])
        _line = line.split('\t')[-1][1:-1].replace("'", "").replace(" ", ",") + ","
        convs.append(_line.split(','))
        #print(convs)
    return convs


def gather_dataset(convs, id2line):
    q = []
    a= []
    total =[]
    global sen_length
    for conv in convs:
        if conv[-1] == "":
            conv = conv[:-1]


        for i in range(len(conv)):
            cleaned = clean(id2line[conv[i]])  # string
            inlist = cleaned.split(" ")
            shortenough = False
            if i < len(conv) - 1: # questions
                cleaned2 = clean(id2line[conv[i+1]])
                inlist2 = cleaned2.split(" ")
                if len(inlist) < sen_length and len(inlist2) < sen_length :
                    q.append(cleaned)
                    shortenough = True

            if i > 0:  # answers
                cleaned2 = clean(id2line[conv[i-1]])
                inlist2 = cleaned2.split(" ")
                if len(inlist) < sen_length and len(inlist2) < sen_length:
                    a.append(cleaned)
                    shortenough = True


            if shortenough:
                total.append(cleaned)



    return total, q, a
    #return dialogues

def clean(line):
    #line = line.lower()
    line = line.replace("\'re", " are ").replace("\'ll", " will ").replace("\'m", " am ").replace("?", " ? ").replace("!", " ! ")
    line = re.sub(r"[^A-Za-z0-9!?]", " ", line)
    line = re.sub("\s+", " ", line)  # 연속되는 공백문자 공백하나로 통일
    line = line.strip()
    return line



def prepare_seq2seq_files(total, q, a, path=''):
    # open files

    global sen_length
    title0 = './extracted_data/movie_dialogue_' + str(sen_length) + '_T.txt'
    title1 = './extracted_data/movie_dialogue_'+str(sen_length) + '_Q.txt'
    title2 = './extracted_data/movie_dialogue_' + str(sen_length) + '_A.txt'
    T = open(path + title0, 'w', encoding='utf8')
    Q = open(path + title1,'w', encoding='utf8')
    A = open(path + title2,'w', encoding='utf8')
    #movie_dialogue = open(path + title,'w', encoding='utf8')

    for i in range(len(total)):
        #print(dialogue[i])
        T.write(total[i] + " <eos>\n ")

    for i in range(len(q)):
        #print(dialogue[i])
        Q.write(q[i] + " <eos>\n ")
        A.write(a[i] + " <eos>\n ")
        if i % 10000 == 0:
            print('\nwritten %d lines'%(i))
    T.close()
    Q.close()
    A.close()



id2line = get_id2line()
convs = get_conversations()
#dialogue = gather_dataset(convs, id2line)#questions, answers = gather_dataset(convs, id2line)
total , questions, answers = gather_dataset(convs, id2line)
prepare_seq2seq_files(total, questions, answers)
print('everything done!')