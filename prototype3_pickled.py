# A method to call the pickled machine learning model for pos/neg sentiment analysis of songs' lyrics.
# Author: VM based on an online course by sentdex.
# Course Webpage: https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/
# GitHub: https://github.com/PythonProgramming/NLTK-3----Natural-Language-Processing-with-Python-series

import nltk
import random
# from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


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


documents_f = open("pickled_classifiers/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("pickled_classifiers/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# featuresets_f = open("pickled_classifiers/featuresets.pickle", "rb")
# featuresets = pickle.load(featuresets_f)
# featuresets_f.close()
#
# random.shuffle(featuresets)
# print(len(featuresets))
#
# testing_set = featuresets[755:]
# training_set = featuresets[:755]


open_file = open("pickled_classifiers/bayes_classifier.pickle", "rb")
bayes_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_classifiers/mnb_classifier.pickle", "rb")
mnb_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_classifiers/bernoulli_nb_classifier.pickle", "rb")
bernoulli_nb_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_classifiers/logistic_regression_classifier.pickle", "rb")
logistic_regression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_classifiers/sgdc_classifier.pickle", "rb")
sgdc_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_classifiers/linear_svc_classifier.pickle", "rb")
linear_svc_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_classifiers/nu_svc_classifier.pickle", "rb")
nu_svc_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(bayes_classifier, mnb_classifier, bernoulli_nb_classifier,
                                  logistic_regression_classifier, sgdc_classifier, linear_svc_classifier,
                                  nu_svc_classifier)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


# open_file = open("test_trained.pkl", "rb")
# trained_classifier = pickle.load(open_file)
# open_file.close()

#
# def sentiment_trained(text):
#     feats = find_features(text)
#     return trained_classifier.classify(feats),trained_classifier.confidence(feats)


moby = "It might seem crazy what I'm 'bout to say Sunshine she's here, you can take a break I'm a hot air balloon that could go to space With the air, like I don't care baby by the way Huh, because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Here come bad news, talking this and that (Yeah) Well, give me all you got, and don't hold it back (Yeah) Well, I should probably warn you I'll be just fine (Yeah) No offense to you, don't waste your time Here's why Because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Hey, come on, uh Bring me down, can't nuthin' (happy) Bring me down My level is too high to bring me down (happy) Can't nuthin', bring me down (happy) I said, let me tell you now, unh (happy) Bring me down, can't nuthin', bring me down (happy, happy, happy) My level is too high to bring me down (happy, happy, happy) Can't nuthin' bring me down (happy, happy, happy) I said Because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Come on, unh bring me down can't nuthin' (happy, happy, happy) Bring me down my level is too high (happy, happy, happy) Bring me down can't nuthin' (happy, happy, happy) Bring me down, I said Because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you, eh eh eh Because I'm happy Clap along if you feel like that's what you wanna do Because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you, eh hey Because I'm happy Clap along if you feel like that's what you wanna do, heh come on"
imm = "I'm so tired of being here Suppressed by all my childish fears And if you have to leave I wish that you would just leave 'Cause your presence still lingers here And it won't leave me alone These wounds won't seem to heal, this pain is just too real There's just too much that time cannot erase When you cried, I'd wipe away all of your tears When you'd scream, I'd fight away all of your fears And I held your hand through all of these years But you still have all of me You used to captivate me by your resonating light Now, I'm bound by the life you left behind Your face it haunts my once pleasant dreams Your voice it chased away all the sanity in me These wounds won't seem to heal, this pain is just too real There's just too much that time cannot erase When you cried, I'd wipe away all of your tears When you'd scream, I'd fight away all of your fears And I held your hand through all of these years But you still have all of me I've tried so hard to tell myself that you're gone But though you're still with me, I've been alone all along When you cried, I'd wipe away all of your tears When you'd scream, I'd fight away all of your fears And I held your hand through all of these years You still have all of me, me, me"
print(sentiment(imm))
# print(sentiment_trained(imm))
