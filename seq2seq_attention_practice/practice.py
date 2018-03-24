
import re
text='ddddddddddd           dddddddddddddddddddfdDDDDDDdf#\" ++!?!000@#@dfdf'
#text = re.sub(r"[^A-Za-z0-9!?]", " ", text)
text = re.sub("\s+", " ",text)
print(text)


s=" dddddd\n\n ddd\n\n"
print(s.strip())

def f():
    return 1,2


