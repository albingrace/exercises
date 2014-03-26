# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 16:07:31 2014

@author: Albingrace
"""

from time import time

from sklearn.feature_extraction.text import CountVectorizer


stop_list = set('for a of the if and to from as in online with at on learn item items month page you your thanks thank com shop'.split())
n_features = 5000 # only consider top 5000 terms according to term frequency


t0 = time()


print('Getting corpus from documents...')
print('top 10%% stop words are ignored and top %d terms are considered...') % n_features

# Tokenize and vectorize the each text document
vectorizer = CountVectorizer(max_df=0.9, stop_words= stop_list, max_features=n_features)

Corpus = vectorizer.fit_transform(line for line in open('deals.txt'))
print("done in %0.3fs." % (time() - t0))

freqs = [(word, Corpus.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
sorted_freqs =  sorted(freqs, key = lambda x: -x[1])


# find the most popular term
most_popular = []
print('\nThe most popular terms are:')
for term,freq in sorted_freqs:
    if freq==sorted_freqs[0][1]:
        most_popular.append(term.encode("utf-8")) # decode from unicode to utf-8
    else:
        break
    
print(most_popular)


#find the least popular term
least_popular = []
print('\nThe least popular terms are:')
for term,freq in reversed(sorted_freqs):
    if freq==sorted_freqs[-1][1]:
        least_popular.append(term.encode("utf-8")) # decode from unicode to utf-8
    else:
        break
    
print(least_popular)


# find how many types of guitar are mentioned

# roughly find out the numbers of documents that contain Guitar
print('\nNumber of Guitar types by roughly computing: %d') % Corpus.getcol(vectorizer.vocabulary_.get('guitar')).sum()

# when I looked into those documents, some of them did not mention any guitar type
# so I tried a different way: extract the word before "guitar" and eliminate some meaningless term
new_documents = [ line for line in open('deals.txt') if 'guitar' in line.lower().split()]
texts = [[word for word in document.lower().split() if word not in stop_list]
         for document in new_documents]

types = []
for line in texts:
    types.append(line[line.index('guitar')-1])

# find union of guitar types
types = set(types)
types = list(types)

# delete terms that contain digit or punctuation letters
final_types = []
for term in types:
    if term.isalpha():
        final_types.append(term)

print('\nNumber of Guitar types by filtering: %d') % len(final_types)
print('They are: %s') % final_types

