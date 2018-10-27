import pandas as pd
import nltk
from nltk.corpus import PlaintextCorpusReader

files = ".*\.txt"

corpus0 = PlaintextCorpusReader("/Users/Shreshth/Documents/du_shman/DM assignment/msc-plagiarism-assigment/", files)
corpus  = nltk.Text(corpus0.words())
print()
#input()

