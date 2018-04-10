import random
import re
sen_length = 10

''' 
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''




def get_id2line():
    lines = open('movie_lines.tsv',encoding='utf8').read()[1:].split('\n')

    #rint(lines[0])
    id2line = {}
    for line in lines:
        _line = line.split('\t')
        #print(_line)
        if len(_line) >= 5: # ==
            #print(_line)
            id2line[_line[0]] = _line[4]
    #print(id2line)
    return id2line


'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''


def get_conversations():
    #a = open('check.txt','w',encoding='utf8')
    #a.close()
    conv_lines = open('movie_conversations.tsv', encoding='utf8').read().split('\n')

    convs = []
    for line in conv_lines[:-1]:
        #print(line)
        #a = open('check.txt', 'a', encoding='utf8')
        #print(line.split('\t')[-1][1:-1])
        _line = line.split('\t')[-1][1:-1].replace("'", "").replace(" ", ",") + ","
        #print(type(_line))
        #print(_line)
        #a.write(_line)
        #a.close()
        convs.append(_line.split(','))
        #print(convs)
    return convs


'''
    1. Get each conversation
    2. Get each line from conversation
    3. Save each conversation to file
'''


def extract_conversations(convs, id2line, path=''):
    idx = 0
    for conv in convs:
        f_conv = open(path + str(idx) + '.txt', 'w')
        for line_id in conv:
            f_conv.write(id2line[line_id])
            f_conv.write('\n')
        f_conv.close()
        idx += 1


'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''


def gather_dataset(convs, id2line):

    dialogues =[]
    global sen_length
    for conv in convs:
        #print(conv)
        if conv[-1] == "":
            conv = conv[:-1]
        toolong = False

        for i in range(len(conv)):
            cleaned = clean(id2line[conv[i]])  #string
            inlist = cleaned.split(" ")
            if len(inlist) > sen_length:
                toolong = True
                break

        if not toolong:
            for i in range(len(conv)):
                cleaned = clean(id2line[conv[i]])  #string
                #print(cleaned)
                dialogues.append(cleaned)


    #return questions, answers
    return dialogues

def clean(line):
    line = line.lower()
    line = line.replace("\'re", " are ").replace("\'ll", " will ").replace("\'m", " am ").replace("?", " ? ").replace("!", " ! ")
    line = re.sub(r"[^a-z0-9!?]", " ", line)
    line = re.sub("\s+", " ", line)  # 연속되는 공백문자 공백하나로 통일
    line = line.strip()
    return line

'''
    We need 4 files
    1. train.enc : Encoder input for training
    2. train.dec : Decoder input for training
    3. test.enc  : Encoder input for testing
    4. test.dec  : Decoder input for testing
'''


def prepare_seq2seq_files(dialogue, path='', TESTSET_SIZE=30000):
    # open files

    global sen_length
    title = 'movie_dialogue_'+str(sen_length) +'.txt'
    movie_dialogue = open(path + title,'w', encoding='utf8')

    #print(len(questions))
    #print(len(answers))
    for i in range(len(dialogue)):
        movie_dialogue.write(dialogue[i] + " <eos>\n ")
        if i % 10000 == 0:
            print('\nwritten %d lines'%(i))

            # close files
    #train_enc.close()
    #train_dec.close()
    #test_enc.close()
    #test_dec.close()
    movie_dialogue.close()


####
#main()
####

id2line = get_id2line()
print( ' gathered id2line dictionary.\n')
convs = get_conversations()
print('>> gathered conversations.\n')
dialogue = gather_dataset(convs, id2line)#questions, answers = gather_dataset(convs, id2line)
print( '총 대화' , len(dialogue))

prepare_seq2seq_files(dialogue)
print('everything done!')