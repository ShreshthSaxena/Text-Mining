import csv
import nltk
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster import hierarchy
from scipy.spatial import distance

import numpy


tokenizer = RegexpTokenizer(r'\w+')
docid=[211,422,734,808,936,202,440,532,606,817,909,1147,321,1349,505,541,743,826,1019, 1037,1046,1138]
doc=[]


stoplist = stopwords.words('english')
additional_stopwords = "google yahoo archie bing search engine internet user crawler launch history engines world wide web download system first data page index file"
stoplist.extend(additional_stopwords.split())

stemmer = PorterStemmer()
for l in docid:
    fileName="ass1-"+str(l)+".txt"
    print(fileName)
    f=open(fileName)
    tokens=[]
    for x in f:
        tokens.extend(tokenizer.tokenize(x))
    token=[stemmer.stem(x.lower()) for x in tokens if x.isalpha()==True and x.lower() not in stoplist]
    print(str(len(token))) 
    doc.append(token) # contains the list of words in each document

lenDoc=len(doc)
with open('data.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(doc)

csvFile.close()

#100-jaccard-> dissimilaity
jaccardMat=numpy.zeros([lenDoc,lenDoc])

for l in range(0,lenDoc):
    for i in range(l,lenDoc):
        intLen=len(list(set(doc[l]).intersection(doc[i])))
        uniLen=len(list(set(doc[l]) | set(doc[i])))
        jaccardMat[l][i]=numpy.around((intLen/uniLen)*100,decimals=2)
        jaccardMat[i][l]=jaccardMat[l][i]


with open('data 1.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(jaccardMat)


#displaying the similarity matrix
#sim_matrix_dataframe = pd.DataFrame(jaccardMat, docid, docid)
#print(sim_matrix_dataframe)

plot.imshow(jaccardMat, interpolation='nearest', cmap=plot.cm.ocean)
plot.xticks(rotation=90)
plot.title("Jaccard Matrix", color='Blue')
plot.xlabel("Document ID", color='green')
plot.ylabel("Document ID", color='green')
plot.xticks(range(len(docid)), docid, fontsize=8, color='blue')
plot.yticks(range(len(docid)), docid, fontsize=8, color='blue')
plot.colorbar()
plot.show()


jaccardMat= np.divide(jaccardMat, 100)
#print(jaccardMat)
disMat = distance.squareform(1-jaccardMat)
#print(disMat)


Z = hierarchy.linkage(disMat, method="complete")
clusters = hierarchy.cut_tree(Z, 4)

#Plottting Dendrogram
fig = plot.figure()
den= hierarchy.dendrogram(Z, labels=docid, color_threshold=0.77,get_leaves=False, orientation='top')
plot.title("Dendrogram")
plot.xlabel("Dissimilarity")
plot.ylabel("Document IDs")
plot.rcParams['lines.linewidth'] = 2
plot.rcParams['lines.color'] = 'r'
plot.show()


#Plottting Dendrogram
fig = plot.figure()
den= hierarchy.dendrogram(Z, labels=docid, color_threshold=0.5,get_leaves=False, orientation='top',count_sort='ascending')
plot.title("Dendrogram")
plot.xlabel("Dissimilarity")
plot.ylabel("Document IDs")
plot.show()

csvFile.close()
