# Based on Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)


# 1. Preparing the data for text processing
#  2. Preprocess the dataset into more convenient format
# 3. Use the bag-of-words model
#  4. Transform words into feature vectors
#  5. Assess word relevancy via term frequency-inverse document frequency
#  6. Clean text data](#Cleaning-text-data)
#  7. Process documents into tokens](#Processing-documents-into-tokens)
# -8. Train a logistic regression model for document classification](#Training-a-logistic-regression-model-for-document-classification)


# # Prepare the data for text processing


import pyprind
import pandas as pd
import os

basepath = '/Users/vasilis/Desktop/Lennon/lyrics_custom_corpus'

labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()
# for s in ('test', 'train'):
for l in ('pos', 'neg'):
    path = os.path.join(basepath, l)
    for file in os.listdir(path):
        with open(os.path.join(path, file),
                  'r', encoding='utf-8') as infile:
            txt = infile.read()
        df = df.append([[txt, labels[l]]],
                       ignore_index=True)
df.columns = ['review', 'sentiment']

# Shuffle DataFrame:

import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
# print(df)

# Save the assembled data as CSV file

df.to_csv('lyrics_data.csv', index=False, encoding='utf-8')

import pandas as pd

# df = pd.read_csv('lyrics_data.csv', encoding='utf-8')
# df.head(3)


# Call the fit_transform method on CountVectorizer to construct the vocabulary of the bag-of-words model.

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

# print(count.vocabulary_)
# print(bag.toarray())

# Assess word relevancy via term frequency-inverse document frequency

np.set_printoptions(precision=2)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs))
      .toarray())

tf_is = 3
n_docs = 3
idf_is = np.log((n_docs + 1) / (3 + 1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)

tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf

l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf ** 2))
l2_tfidf

# Cleaning text data


df.loc[0, 'review'][-50:]

import re


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


preprocessor(df.loc[0, 'review'][-50:])

preprocessor("</a>This :) is :( a test :-)!")

df['review'] = df['review'].apply(preprocessor)

# print(df)

# Processing documents into tokens

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
 if w not in stop]

# Train a logistic regression model for document classification

# Strip HTML and punctuation to speed up the GridSearch later:

X_train = df.loc[:755, 'review'].values
y_train = df.loc[:755, 'sentiment'].values
X_test = df.loc[755:, 'review'].values
y_test = df.loc[755:, 'sentiment'].values

# Fine tuning LogisticRegression


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1, return_train_score=True)

gs_lr_tfidf.fit(X_train, y_train)

print('LogisticRegression: Best  parameter set: %s ' % gs_lr_tfidf.best_params_)
print('LogisticRegression: CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('LogisticRegression: Test Accuracy: %.3f' % clf.score(X_test, y_test))

import matplotlib.pyplot as plt

# plt.plot(k_range, grid_mean_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated Accuracy')

results_grid = pd.DataFrame(gs_lr_tfidf.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

print(results_grid)

# Write outputs to Excel.

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np

writer = ExcelWriter('LogisticRegressionTune.xlsx')
results_grid.to_excel(writer, 'Sheet1', index=False)
writer.save()
