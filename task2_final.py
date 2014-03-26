""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""

"""
@author: Qian Wan
"""

from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn import decomposition


txtfile = './data/deals.txt'
n_features = 2000
n_topics = 10 # number of topics
n_top_words = 5 # n words per topic
n_groups = 10# number of groups

# set some stop words that appear frequently but with less meaning
stop_list = set('for a of the if and to from as in online with at on learn item items month page you your thanks thank com shop'.split())

t0 = time()

print('Getting corpus from documents...')
print('top 20%% stop words are ignored and top %d terms are considered...') % n_features

vectorizer = CountVectorizer(max_df=0.8, stop_words= stop_list, max_features=n_features)
Corpus = vectorizer.fit_transform(line for line in open(txtfile))
print("done in %0.3fs." % (time() - t0))

# give weights to each entry by term-frequency nverse document-frequency method
print('TF-IDF weighting...')
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(Corpus)
print("done in %0.3fs.\n" % (time() - t0))


# Non Negative Matrix Factorization of the term frequency matrix
# Fit the NMF model
print("Fitting the Negative Matrix Factorization model... this will take about 1 to 2 minutes...")
nmf = decomposition.NMF(n_components=n_topics).fit(tfidf)
print("done in %0.3fs.\n" % (time() - t0))


# Inverse the vectorizer vocabulary
feature_names = vectorizer.get_feature_names()

print('%d topics are selected, they are:') % n_topics
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % (topic_idx+1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))


# I don't quite get the difference between "group" and "topic",
# so I implemented a K-means based clustering by setting number of groups the same as topic's

## K-means clustering for grouping
print('\nGrouping using K means, K=%d...') % n_groups
km = KMeans(n_clusters=n_groups, init='k-means++', max_iter=100, n_init=5)
km.fit(tfidf)
labels = km.labels_
print('%d groups have been formed as "labels"') % n_groups


# Thanks for reviewing

