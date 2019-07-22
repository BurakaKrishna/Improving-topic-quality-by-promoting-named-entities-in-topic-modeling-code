# Implementation by using tfidf example from wikipedia

import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
dataset = [["this", "is" ,"a","a" ,"sample"],["this","this","this","is","another","example"]]
dct = Dictionary(dataset)  # fit dictionary
corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format
model = TfidfModel(corpus)  #fit model
vector = model[corpus]  # apply model to the first corpus document


def get_max_tf(doc):
    return max(doc,key=itemgetter(1))[1]

entity_list = ["sample","another"]

from operator import itemgetter
def get_max_frequency_weights(doc):
    return max(doc,key=itemgetter(1))[1]

def check_entity(word):
    if dct[word[0]] in entity_list:
        return True
    return False

def upgrade_entity_weights(doc,max_weight):
#     print (max_weight)
    for word_no,word in enumerate(doc):
        if check_entity(word):
            doc[word_no] = (doc[word_no][0],doc[word_no][1]+max_weight)
    return doc

for doc_no,doc in enumerate(corpus):
#     doc_entities = [(0,2),(1,3),(2,1)]
    print (corpus[doc_no])
    max_weight = get_max_frequency_weights(corpus[doc_no])
    print (max_weight)
    corpus[doc_no] = upgrade_entity_weights(doc,max_weight)

#Use this corpus to train lda but giving weights to entities
