

def isnum(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


original = open('misaeng (1).srt', 'r', encoding='utf-8')
cleansed = open('misaeng1.txt', 'w', encoding='UTF_8')


# <i> <i/> 뭔지 몰라서 일단 가만히 내비 둠.
string_saved=''
string = original.readline()
string = string.replace('\ufeff' , '')
while string: # if '-->' exists, below it are dailogues ,  if '- ' exists, they are spoken by different person

    if(isnum(string) or  '-->' in string ):
        string_saved =''

    else:
        while(not isnum(string) and string) :
            string_saved = string_saved.strip() + ' ' +  string.strip()
            string = original.readline()


    cleansed.write(string_saved.strip() + '\n')
    string = original.readline()


    #list1=[]
    #list1.append(string_saved)
    #print(list1)

cleansed.close()



