import nltk
import string
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD as SVD


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

lsa = SVD(n_components = 4, n_iter =100)
doc_top=lsa.fit_transform(tfs)
doc_top=Normalizer(copy=False).fit_transform(doc_top)
terms = tfidf.get_feature_names()
for i, comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    sortedTerms = sorted(termsInComp, key=lambda x:x[1], reverse=True) [:5]
    print ("Topic %d:" %i)
    for term in sortedTerms:
        print (term[0])
    print (" ")


##import umap
##X_topics = lsa.fit_transform(tfs)
##embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)
##
##plt.figure(figsize=(7,5))
##plt.scatter(embedding[:, 0], embedding[:, 1], 
##c = tfidf.get_feature_names(),
##s = 10, # size
##edgecolor='none'
##)
##plt.show()
