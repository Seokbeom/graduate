

def isnum(s):
  try:
    int(s)
    return True
  except ValueError:
    return False

def simplyfi(string):
    if '<i>' in string:  # <i>는 나레이션 같아서 삭제

        string = string.replace('<i>', '')

    if '</i>' in string:  # <i>는 나레이션 같아서 삭제
        # print(string)
        string = string.replace('</i>', '')
        # print(string)
    if ' - ' in string:
        # print(string)
        string = string.replace(' - ', '\n')
    if '- ' in string:
        # print(string)
        string = string.replace('- ', '')

    string = string.replace('.', ' . ')
    string = string.replace('?', ' ? ')
    string = string.replace('!', ' ! ')
    string = string.replace('\'', ' \' ')
    string = string.replace('…', ' … ')
    string = string.replace(',', ' , ')
    string = string.replace('\"' , ' \" ')
    return string.lower()



cleansed = open('cleansed.txt', 'w', encoding='utf-8')
number=1
end =20
while number<=end:

    filename = 'misaeng (%d).srt'%(number)
    original = open(filename, 'rt', encoding='UTF8')

    # <i> <i/> 뭔지 몰라서 일단 가만히 내비 둠. 지워야 할듯
    # - 따로 빼는 작업 필요
    string_saved=''
    string = original.readline()
    string = string.replace('\ufeff' , '')
    while string:
        if isnum(string) or  '-->' in string  or '[' in string or ']' in string or 'not yet alive' in string: #숫자(씬 넘버)이거나 -->가 포함되거나(자막 시간데이터) [가 포함되면(대화가 아님) 건너 뜀
            string_saved =''


        else:

            while(not isnum(string) and string) :
                string_saved = string_saved.strip() + ' ' +  string.strip()
                string = original.readline()
                string = simplyfi(string)
        if len(string_saved) >1:
            cleansed.write(string_saved.strip() + '\n')

        string = original.readline()
        string = simplyfi(string)

    number +=1


cleansed.close()



