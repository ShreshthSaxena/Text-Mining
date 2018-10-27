import nltk
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

path = './msc-plagiarism-assigment/*.txt'   
files=glob.glob(path)
n = len(files)
i=0
fname = [None]*n
raw = [None]*n
ftext = [None]*n
i = 0
for f in files:
        try:
                fname[i] = open(f,"r")
                raw[i] = fname[i].read()
        except UnicodeDecodeError as e:
                fname[i] = open(f,"r",encoding="utf8")        
                raw[i] = fname[i].read()
        ftext[i] = re.sub("[^A-Za-z]", " ", raw[i])
        i = i + 1

tokens = [None]*n
swr = [None]*n
stopwords = stopwords.words("english")
add = ['search','engine','web','internet']
stopwords.extend(add)
st = [None]*n
for i in range(0,n):
    tokens[i] = nltk.word_tokenize(ftext[i])
    swr[i] = [w for w in tokens[i] if not w.lower() in stopwords]
    ps = PorterStemmer()
    st[i]= []
    for ws in swr[i]:
        st[i].append(ps.stem(ws))

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

stem_words = [None]*n
for i in range(0,n):
    stem_words[i] = set(st[i])

jam = np.zeros((n,n),dtype='double')
min=1
max=0
for i in range (0,n):
    for j in range(i,n):
        jam[i][j] = jaccard_similarity(st[i], st[j])
        jam[j][i] = jam[i][j]
        
np.set_printoptions(precision=3)
print(jam)

#DENDROGRAM

plt.figure(figsize=(10, 7))  
plt.title("hist")  

distanceMatrix = 1-jam
print(distanceMatrix)
dend = dendrogram(linkage(1-jam,method='complete'), 5,
           color_threshold=10, 
           leaf_font_size=10)
plt.show()

cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='complete')  
cluster.fit_predict(1-jam.T)
