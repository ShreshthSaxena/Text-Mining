import nltk
import string
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

path = './msc-plagiarism-assigment'
token_dict = {}


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

for dirpath, dirs, files in os.walk(path):
    for f in files:
        fname = os.path.join(dirpath, f)
        print ("fname=", fname)
        try:
            with open(fname) as pearl:
                text = pearl.read()
                token_dict[f] = re.sub("[^A-Za-z]", " ", text)
        except UnicodeDecodeError as e:
            with open(fname,encoding="utf8") as pearl:
                text = pearl.read()
                token_dict[f] = re.sub("[^A-Za-z]", " ", text)
        

stopwords = stopwords.words("english")
add = ['search','engine','web','internet']
stopwords.extend(add)
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=stopwords)
tfs = tfidf.fit_transform(token_dict.values())

cosim = cosine_similarity(tfs,tfs)
print(cosim)


#Spectral Clustering
#from sklearn.cluster import SpectralClustering
#sc = SpectralClustering(n_clusters = 2, affinity = 'precomputed')
#sc.fit_predict(cosim[0:3,0:3])

#DBScan
#from sklearn.cluster import DBSCAN
#db=DBSCAN(min_samples=1)
#db.fit_predict(cosim)

#K-Means

from sklearn.cluster import KMeans

#sse={}

eigen_values, eigen_vectors = np.linalg.eigh(cosim)
km = KMeans(n_clusters=5, init='k-means++')
km.fit_predict(eigen_vectors[:, -4:])

#for i in range(1,21):   
#    km = KMeans(n_clusters=i, init='k-means++')
#    km.fit_predict(eigen_vectors[:, -4:])
#    sse[i]=km.inertia_
#x,y=(zip(*sse.items()))
#plt.plot(x,y)
#plt.title("Elbow curve")
#plt.xlabel("Clusters(k)")
#plt.ylabel("SSE")
#plt.show()

#dense_tfs = tfs.toarray()
#KM = KMeans(n_clusters=i, n_init=50, max_iter=100)
#KM.fit_transform(dense_tfs)

#HIERARCHICAL


from scipy.cluster.hierarchy import dendrogram, linkage,cut_tree
from sklearn.cluster import AgglomerativeClustering


plt.figure(figsize=(10, 7))  
plt.title("dendrogram")  
distanceMatrix = 1-cosim
Z=linkage(cosim,method='complete')

dend = dendrogram(Z, 5, orientation = 'top', 
           color_threshold=10,
           leaf_font_size=10, show_leaf_counts=True)
plt.show()

cluster = AgglomerativeClustering(n_clusters=10, affinity='precomputed', linkage='complete')  
cluster.fit_predict(cosim)
#str = 'all great and precious things are lonely.'
#response = tfidf.transform([str])
#print (response)

#feature_names = tfidf.get_feature_names()
#for col in response.nonzero()[1]:
#    print (feature_names[col], ' - ', response[0, col])

