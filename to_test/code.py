import pandas as pd
import nltk
from nltk.corpus import stopwords

f=open('ass1_422.txt','rU')
raw=f.read()
tokens = nltk.word_tokenize(raw)
print(tokens)

stopwords = stopwords.words("english")
add = ['search','engine','web','.','(',')',',','!','&','?',':',';']
stopwords.extend(add)
sw = []
for w in tokens:
	if w not in stopwords:
		sw.append(w)

print(sw)


