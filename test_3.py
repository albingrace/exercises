# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 16:30:16 2014

@author: Qian Wan
"""

from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import RidgeClassifier
#from sklearn import cross_validation
#from sklearn import metrics
#from sklearn.feature_selection import SelectKBest, chi2
import fileinput
import numpy as np
import math

test_txtfile = 'test_deals.txt'

# combine two deals for the purpose of having common features
filenames = ['good_deals.txt', 'bad_deals.txt']
outfilename = 'combined.txt'
with open(outfilename, 'w') as fout:
    for line in fileinput.input(filenames):
        fout.write(line)

# set some stop words that appear frequently but with less meaning
stop_list = set('for a of the if and to from as in online with at on learn item items month page you your thanks thank com shop'.split())


t0 = time()


print('Getting the corpus from good_deals and bad_deals...')
print('top 10% stop words are ignored...') 
vectorizer = CountVectorizer(max_df=0.9, stop_words= stop_list)

# generate training matrix with word features and their frequency, each text line is one observation
Corpus_train = vectorizer.fit_transform(line for line in open(outfilename))
print("done in %0.3fs." % (time() - t0))

# generate testing matrix by using a same vectorizer
Corpus_test = vectorizer.transform(line for line in open(test_txtfile))

# if consider all good deals as coupon-required deals
#n = Corpus.shape[0]/2
#Categories =np.append( np.tile(1,int(n)), np.tile(0,int(n)))


# if only for those which contains a exact coupon code
Y_train = np.tile(0,int(Corpus_train.shape[0]))
Y_test = np.tile(0,int(Corpus_test.shape[0]))

# create binary categories, 1 means the deal requires a coupon code, 0 means not
# the 14th and 26th deals require a coupon code in the training good deals
Y_train[13]=1
Y_train[25]=1

# the 6th, 32nd, 33rd and 49th deals require a coupon code in the testing deals
Y_test[5]=1
Y_test[31]=1
Y_test[32]=1
Y_test[49]=1


# feature selection, chi2 is fit for sparse data
# actually, here the number of features is not very large and feature selection is not very neccesary 

## select top 100 ranked features, by tuning, this is the best number of top featues that can represent a suitable portion of total features
#ch2 = SelectKBest(chi2, k=100)  # too few features may lead to insufficient fitting
#Corpus_train = ch2.fit_transform(Corpus_train, Y_train)
#Corpus_test = ch2.transform(Corpus_test)


# define a function to print out classification rate
n_test = Corpus_test.shape[0]
#cv = cross_validation.KFold(n_test, n_folds=10) # 10-fold cross validation

f= open(test_txtfile)
def print_rate(y_test, y_pred, n_test, method_name):   
    incorrect_num = sum([int(math.fabs(a - b)) for a, b in zip(y_test, y_pred)])
    print('\n%s: correct classification rate is %d/%d') % (method_name, n_test - incorrect_num, n_test)
    print('While there are %d predicted coupon-required deals...\n') % sum(y_pred)
    if sum(y_pred):
        print('They are:')
        for i in range(0,n_test-1):
            deal = f.readline()
            if y_pred[i]:
                print('%s') % deal
    

def Predict():
    print('\nThere are %d new deals') % n_test

    # Using the KNN classifier
    clf_KNN = KNeighborsClassifier(n_neighbors=3) # KNN doesnot work even if k has been tuned
    #clf_KNN = KNeighborsClassifier(n_neighbors=7)
    #clf_KNN = KNeighborsClassifier(n_neighbors=11)
    clf_KNN.fit(Corpus_train, Y_train)
    Y_pred_KNN = clf_KNN.predict(Corpus_test)
    print_rate(Y_test, Y_pred_KNN, n_test, 'KNNClassifier')
    
    # Using the SVM classifier
    clf_SVM = svm.SVC()
    clf_SVM.fit(Corpus_train, Y_train)
    Y_pred_SVM = clf_SVM.predict(Corpus_test)
    print_rate(Y_test, Y_pred_SVM, n_test, 'SVMClassifier')
    
    # Using the Ridge classifier
    clf_RC = RidgeClassifier(tol=0.01, solver="lsqr")
    #clf_RC = RidgeClassifier(tol=0.1, solver="lsqr")
    clf_RC.fit(Corpus_train, Y_train)
    Y_pred_RC = clf_RC.predict(Corpus_test)
    print_rate(Y_test, Y_pred_RC, n_test, 'RidgeClassifier')
    
    # won't consider Random Forests or Decision Trees beacause they work bad for high sparse dimensions
    
    
    # Using the Multinomial Naive Bayes classifier
    # I expect that this MNB classifier will do the best since it is designed for occurrence counts features
    #clf_MNB = MultinomialNB(alpha=0.01) #smoothing parameter = 0.01 is worse than 0.1
    clf_MNB = MultinomialNB(alpha=0.1)
    #clf_MNB = MultinomialNB(alpha=0.3) #a big smoothing rate doesnot benefit the model
    #clf_MNB = MultinomialNB(alpha=0.2) #or alpha = 0.05 can generate the best outcome
    clf_MNB.fit(Corpus_train, Y_train)
    Y_pred_MNB = clf_MNB.predict(Corpus_test)
    print_rate(Y_test, Y_pred_MNB, n_test, 'MultinomialNBClassifier')
    #score = metrics.f1_score(Y_test, Y_pred_MNB)
    #scores = cross_validation.cross_val_score(clf_MNB, Corpus_train, Y_train, cv=cv)


# Thus, after trying four distinct classifiers, the Multinomial Naive Bayes classifier is the best performer.

# How general is the classifier?
# The Multinomial Naive Bayes classifier built for this problem only takes the
# original text files as the input, while having the prior knowledge on training output.
# For the training deals, we only need to know which one requires a coupon code, then
# for a coming identically formatted new deal, we can simply apply the classifier and get a prediction.


# How was the classifier tested?
# An error rate metric was simply used. I assume I know which deals require coupon codes
# for test_deals, then compared the correctly classified categories across different classifiers.
# Besides, although some models seem to have a high rate, they might have a overfitting problem.
# Hence, I computed the number of predicted coupon-required deals for each model, it turns out
# to be true that KNN, SVM and Ridge classifiers overfitted data.
#
# Cross validation such as 10-fold CV can also be applied, since here a test_deals.txt is provided,
# I simply use it as my testing samples.




