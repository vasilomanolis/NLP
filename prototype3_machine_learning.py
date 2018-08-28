# A machine learning model for pos/neg sentiment analysis of songs' lyrics.
# Author: VM based on an online course by sentdex.
# Course Webpage: https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/
# GitHub: https://github.com/PythonProgramming/NLTK-3----Natural-Language-Processing-with-Python-series

import shutil
import string
from itertools import chain
from nltk import word_tokenize

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier as nbc
from nltk.corpus import CategorizedPlaintextCorpusReader
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.feature_extraction.text import CountVectorizer

import nltk
import random
import re
import pickle
import os


# A class that combines several classifiers via a voting system. Each classifier has an equal vote.
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# Provide path to the custom corpora

mydir = '/Users/vasilis/Desktop/Lennon/lyrics_custom_corpus'

# Read data from our custom corpora

mr = CategorizedPlaintextCorpusReader(mydir, r'(?!\.).*\.txt', cat_pattern=r'(neg|pos)/.*')

# Clean lyrics from the English stop words.
stop = stopwords.words('english')

documents = [(list(mr.words(fileid)), category)
             for category in mr.categories()
             for fileid in mr.fileids(category)]

classifiers_dir = '/Users/vasilis/vxm773/Lennon/pickled_classifiers'

if os.path.exists(classifiers_dir):
    shutil.rmtree(classifiers_dir)
os.makedirs(classifiers_dir)

save_documents = open("pickled_classifiers/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

# Shuffle lyrics in order to avoid training only towards pos/neg lyrics.

random.shuffle(documents)

all_words = []

for w in mr.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

# remove punctuation and stop words
# print("15 most commons")
# print(all_words.most_common(15))
# print(all_words["stupid"])

# Create word_features to check against the most frequent 30000 words
word_features = list(all_words.keys())[:3000]

save_word_features = open("pickled_classifiers/word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    # all the words of documents are included in the set words
    words = set(document)

    # Create a dictionary where the key is the word. Then create a boolean
    # according to if the word is included in the 3000 words.
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# Use a simple tokenizer by splitting the words.
def tokenizer(text):
    return text.split()


# A method to clean lyrics from stopwords and punctuation.
def clean_lyrics(lyrics):
    stop_words = set(stopwords.words('english'))

    # Force lower case on lyrics.
    try:
        lyrics = lyrics.lower()
    except:
        lyrics = lyrics
    # Tokenize lyrics.
    word_tokens = word_tokenize(lyrics)
    # word_tokens = tokenizer(lyrics)
    word_token_without_punctuation = [x for x in word_tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]
    # Remove the stop words.
    filtered_lyrics = [w for w in word_token_without_punctuation if not w in stop_words]
    # Return the lyrics without stop words.
    return filtered_lyrics


feature_sets = [(find_features(rev), category) for (rev, category) in documents]

# Split our dataset into two parts: training and testing set
training_set = feature_sets[:755]
testing_set = feature_sets[755:]


# A method to save classifiers using pickle
def pickle_classifiers(classifier, classifier_name):
    path_output = "pickled_classifiers/" + classifier_name + ".pickle"
    save_classifier = open(path_output, "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


# Use Naive Bayes classifier
# Documentation: https://www.nltk.org/_modules/nltk/classify/naivebayes.html

bayes_classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes classifier accuracy %:", (nltk.classify.accuracy(bayes_classifier, testing_set)) * 100)
bayes_classifier.show_most_informative_features(50)

pickle_classifiers(bayes_classifier, "bayes_classifier")

# Use MultinomialNB classifier
# [Naive Bayes classifier for multinomial models]
# The multinomial Naive Bayes classifier is suitable for classification with discrete features.
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

mnb_classifier = SklearnClassifier(MultinomialNB())
mnb_classifier.train(training_set)
print("MultinomialNB accuracy %:", nltk.classify.accuracy(mnb_classifier, testing_set) * 100)

pickle_classifiers(mnb_classifier, "mnb_classifier")

# Use BernoulliNB classifier
# [Naive Bayes classifier for multivariate Bernoulli models]
# Like MultinomialNB, this classifier is suitable for discrete data. The difference is that
# while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features.
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html

bernoulli_nb_classifier = SklearnClassifier(BernoulliNB())
bernoulli_nb_classifier.train(training_set)
print("BernoulliNB accuracy $%", nltk.classify.accuracy(bernoulli_nb_classifier, testing_set) * 100)

pickle_classifiers(bernoulli_nb_classifier, "bernoulli_nb_classifier")

# Use Logistic Regression classifier
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

logistic_regression_classifier = SklearnClassifier(LogisticRegression())
logistic_regression_classifier.train(training_set)
print("Logistic Regression accuracy %:",
      (nltk.classify.accuracy(logistic_regression_classifier, testing_set)) * 100)

pickle_classifiers(logistic_regression_classifier, "logistic_regression_classifier")

# Use fine-tuned Logistic Regression classifier

logistic_regression_classifier_trained = SklearnClassifier(
    LogisticRegression(penalty="l2", C=100, class_weight=None, random_state=None, solver="liblinear", max_iter=100,
                       multi_class="ovr", verbose=0, warm_start=False, n_jobs=1))
logistic_regression_classifier_trained.train(training_set)
print("LogisticRegression_classifier_trained accuracy %:",
      (nltk.classify.accuracy(logistic_regression_classifier_trained, testing_set)) * 100)

pickle_classifiers(logistic_regression_classifier_trained, "logistic_regression_classifier_trained")

# Use SGDClassifier classifier
# Linear classifiers (SVM, logistic regression, a.o.) with SGD training.
# For best results using the default learning rate schedule, the data should have zero mean and unit variance.
# This implementation works with data represented as dense or sparse arrays of floating point values for the features.
# The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM).
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html


sgdc_classifier = SklearnClassifier(SGDClassifier())
sgdc_classifier.train(training_set)
print("SGDClassifier_classifier accuracy %:",
      (nltk.classify.accuracy(sgdc_classifier, testing_set)) * 100)

pickle_classifiers(sgdc_classifier, "sgdc_classifier")

# Use SVC_classifier classifier
# C-Support Vector Classification. The implementation is based on libsvm.
# The fit time complexity is more than quadratic with the number of samples which makes
# it hard to scale to dataset with more than a couple of 10000 samples.
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

svc_classifier = SklearnClassifier(SVC())
svc_classifier.train(training_set)
print("SVC_classifier accuracy %:", (nltk.classify.accuracy(svc_classifier, testing_set)) * 100)

pickle_classifiers(svc_classifier, "svc_classifier")

# Use LinearSVC classifier
# Linear Support Vector Classification. Similar to SVC with parameter kernel=’linear’,
# but implemented in terms of liblinear rather than libsvm, so it has more flexibility
# in the choice of penalties and loss functions and should scale better to large numbers of samples.
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

linear_svc_classifier = SklearnClassifier(LinearSVC())
linear_svc_classifier.train(training_set)
print("LinearSVC_classifier accuracy %:", (nltk.classify.accuracy(linear_svc_classifier, testing_set)) * 100)

pickle_classifiers(linear_svc_classifier, "linear_svc_classifier")

# Use NuSVC classifier
# Nu-Support Vector Classification.
# Similar to SVC but uses a parameter to control the number of support vectors.
# The implementation is based on libsvm.
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html

nu_svc_classifier = SklearnClassifier(NuSVC())
nu_svc_classifier.train(training_set)
print("NuSVC_classifier accuracy %:", (nltk.classify.accuracy(nu_svc_classifier, testing_set)) * 100)

pickle_classifiers(nu_svc_classifier, "nu_svc_classifier")

# Create a classifier based on equal votes.
voted_classifier = VoteClassifier(bayes_classifier, mnb_classifier, bernoulli_nb_classifier,
                                  logistic_regression_classifier, sgdc_classifier, linear_svc_classifier,
                                  nu_svc_classifier)

print("Voted_classifier accuracy %:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)
