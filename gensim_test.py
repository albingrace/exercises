# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 22:16:19 2014

@author: Qian Wan
"""
import re
from gensim import corpora

#documents = [ "Online Fingerstyle Guitar Lessons online",
#              "Online Country Guitar Lessons",
#              "Online Rock Guitar Lessons",
#              "Learn Classical Guitar Online",
#              "Espresso Frames Now 40% Off from PictureFrames.com!",
#              "destinations! Starting from only $61/night. Shop Now!",
#              "The EPS user interface management system",
#              "System and human system engineering testing of EPS",
#              "Relation of user perceived response time to error measurement",
#              "The generation of random binary unordered trees",
#              "The intersection graph of paths in trees",
#              "Graph minors IV Widths of trees and well quasi ordering",
#              "Graph minors A survey"]
#              

stoplist = set('for a of the and to in is an online on all at with our your'.split())

# collect statistics about all tokens
dictionary = corpora.Dictionary(re.sub(r'[-!\/?.&]', '', line).lower().split() for line in open('good_deals.txt'))
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]

# remove stop words and words that appear only once
dictionary.filter_tokens(stop_ids + once_ids) 

# keep top keep_n tokens
dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=9)

dictionary.compactify() # remove gaps in id sequence after words that were removed
dictionary.save('/tmp/good_deals.dict')

print(dictionary.token2id)

#print(dictionary.token2id)

class MyCorpus(object):
     def __iter__(self):
         for line in open('good_deals.txt'):
             # assume there's one document per line, tokens separated by whitespace
             yield dictionary.doc2bow(re.sub(r'[-!\/?.&]', '', line).lower().split())

corpus_memory_friendly = MyCorpus() # doesn't load the corpus into memory!
corpora.MmCorpus.serialize('/tmp/good_deals.mm', corpus_memory_friendly)
print(corpus_memory_friendly)



# find types of guitars


#for vector in corpus_memory_friendly: # load one vector into memory at a time
#     print(vector)

#texts = [[word for word in document.lower().split() if word not in stoplist]
#         for document in documents]
#         
#
## remove words that appear only once
#all_tokens = sum(texts)
#tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
#texts = [[word for word in text if word not in tokens_once]
#          for text in texts]
##
##print(texts)
#
#
#dictionary = corpora.Dictionary(texts)
#print(dictionary)
#print(dictionary.token2id)
#
## convert strings to vectors
#corpus = [dictionary.doc2bow(text) for text in texts]
#
#print(corpus)

