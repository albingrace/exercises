# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 15:22:29 2014

@author: Qian Wan
"""

from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn import decomposition

n_features = 2000
n_topics = 10 # number of topics
n_clusters = 10# number of groups
n_top_words = 5 # n words per topic

t0 = time()


# set some stop words that appear frequently but with less meaning
stop_list = set('for a of the if and to from as in online with at on learn item items month page you your thanks thank com shop'.split())
vectorizer = CountVectorizer(max_df=0.8, stop_words= stop_list, max_features=n_features)
Corpus = vectorizer.fit_transform(line for line in open('deals.txt'))

print('TF-IDF weighting...')
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(Corpus)
print("done in %0.3fs.\n" % (time() - t0))


# Fit the NMF model
print("Fitting the NMF model... this will take about one minute...")
nmf = decomposition.NMF(n_components=n_topics).fit(tfidf)
print("done in %0.3fs.\n" % (time() - t0))


# Inverse the vectorizer vocabulary to be able
feature_names = vectorizer.get_feature_names()

print('%d topics are selected, they are:') % n_topics
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))


# I don't quite get the difference between "group" and "topic",
# so I implemented a K-means based clustering by setting number of groups the same as topic's

## K-means clustering for grouping
km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=5)
km.fit(tfidf)
labels = km.labels_




