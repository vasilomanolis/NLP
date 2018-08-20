# based on the pythonprogramming.net tutorial

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

import nltk
import random
import re
import pickle

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


mydir = '/Users/vasilis/Desktop/Lennon/lyrics_custom_corpus'

mr = CategorizedPlaintextCorpusReader(mydir, r'(?!\.).*\.txt', cat_pattern=r'(neg|pos)/.*')
# mr = CategorizedPlaintextCorpusReader(mydir, r'(?!\.).*\.txt', cat_pattern=r'(neg|pos)/.*', encoding='ascii')

stop = stopwords.words('english')
# documents = [([w for w in mr.words(i) if w.lower() not in stop and w.lower() not in string.punctuation], i.split('/')[0]) for i in mr.fileids()]

documents = [(list(mr.words(fileid)), category)
             for category in mr.categories()
             for fileid in mr.fileids(category)]

random.shuffle(documents)



all_words = []

for w in mr.words():

    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# remove punctuation and stop words
# print("15 most commons")
# print(all_words.most_common(15))
# print(all_words["stupid"])

# we check against the most frequent 30000 words
word_features = list(all_words.keys())[:3000]


def find_features(document):
    # all the words of documents are included in the set words
    words = set(document)

    # create empty dictionary
    features = {}
    # the key in the features dictionary is the word. Then it creates a boolean
    # according to if the word is included in the 3000 words
    for w in word_features:
        features[w] = (w in words)

    return features


# A method that accepts the lyrics of a song and returns the lyrics after removing the English stop words.
def clean_lyrics(lyrics):
    stop_words = set(stopwords.words('english'))

    # Force lower case on lyrics.
    try:
        lyrics = lyrics.lower()
    except:
        lyrics = lyrics
    # Tokenize lyrics.
    word_tokens = word_tokenize(lyrics)


    word_token_without_punctuation = [x for x in word_tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]

    # Remove the stp words.
    filtered_lyrics = [w for w in word_token_without_punctuation if not w in stop_words]

    # Return the lyrics without stop words.
    return filtered_lyrics

#print((find_features(mr.words("neg/ML46-Cyanide.txt"))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]


training_set = featuresets[:800]
testing_set = featuresets[800:]

BAYES_classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(BAYES_classifier, testing_set)) * 100)
BAYES_classifier.show_most_informative_features(50)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:", nltk.classify.accuracy(MNB_classifier, testing_set))

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:", nltk.classify.accuracy(BernoulliNB_classifier, testing_set))

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:",
      (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(BAYES_classifier, save_classifier)
# save_classifier.close()


voted_classifier = VoteClassifier(BAYES_classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


print("Classification:", voted_classifier.classify(sad), "Confidence %:",voted_classifier.confidence(sad)*100)
print("Classification:", voted_classifier.classify(happy), "Confidence %:",voted_classifier.confidence(happy)*100)


print("stop")


print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)

