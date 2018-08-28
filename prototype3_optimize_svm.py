from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pylab

# coding: utf-8

# Based on Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)




# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*


# ### Overview

# - [Preparing the IMDb movie review data for text processing](#Preparing-the-IMDb-movie-review-data-for-text-processing)
#   - [Obtaining the IMDb movie review dataset](#Obtaining-the-IMDb-movie-review-dataset)
#   - [Preprocessing the movie dataset into more convenient format](#Preprocessing-the-movie-dataset-into-more-convenient-format)
# - [Introducing the bag-of-words model](#Introducing-the-bag-of-words-model)
#   - [Transforming words into feature vectors](#Transforming-words-into-feature-vectors)
#   - [Assessing word relevancy via term frequency-inverse document frequency](#Assessing-word-relevancy-via-term-frequency-inverse-document-frequency)
#   - [Cleaning text data](#Cleaning-text-data)
#   - [Processing documents into tokens](#Processing-documents-into-tokens)
# - [Training a logistic regression model for document classification](#Training-a-logistic-regression-model-for-document-classification)
# - [Working with bigger data – online algorithms and out-of-core learning](#Working-with-bigger-data-–-online-algorithms-and-out-of-core-learning)
# - [Topic modeling](#Topic-modeling)
#   - [Decomposing text documents with Latent Dirichlet Allocation](#Decomposing-text-documents-with-Latent-Dirichlet-Allocation)
#   - [Latent Dirichlet Allocation with scikit-learn](#Latent-Dirichlet-Allocation-with-scikit-learn)
# - [Summary](#Summary)


# # Preparing the IMDb movie review data for text processing

# ## Obtaining the IMDb movie review dataset

# The IMDB movie review set can be downloaded from [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/).
# After downloading the dataset, decompress the files.
#
# A) If you are working with Linux or MacOS X, open a new terminal windowm `cd` into the download directory and execute
#
# `tar -zxf aclImdb_v1.tar.gz`
#
# B) If you are working with Windows, download an archiver such as [7Zip](http://www.7-zip.org) to extract the files from the download archive.

# **Optional code to download and unzip the dataset via Python:**

# In[2]:
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score


import pyprind
import pandas as pd
import os

# change the `basepath` to the directory of the
# unzipped movie dataset

basepath = '/Users/vasilis/Desktop/Lennon/lyrics_custom_corpus'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
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
            pbar.update()
df.columns = ['review', 'sentiment']

#print(df)
# Shuffling the DataFrame:

# In[24]:


import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
print(df)

# Optional: Saving the assembled data as CSV file:

# In[25]:


df.to_csv('lyrics_data.csv', index=False, encoding='utf-8')

# In[26]:


import pandas as pd

# df = pd.read_csv('lyrics_data.csv', encoding='utf-8')
# df.head(3)

# <hr>
# ### Note
#
# If you have problems with creating the `movie_data.csv` file in the previous chapter, you can find a download a zip archive at
# https://github.com/rasbt/python-machine-learning-book-2nd-edition/tree/master/code/ch08/
# <hr>


# # Introducing the bag-of-words model

# ...

# ## Transforming documents into feature vectors

# By calling the fit_transform method on CountVectorizer, we just constructed the vocabulary of the bag-of-words model and transformed the following three sentences into sparse feature vectors:
# 1. The sun is shining
# 2. The weather is sweet
# 3. The sun is shining, the weather is sweet, and one and one is two
#

# In[6]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

# Now let us print the contents of the vocabulary to get a better understanding of the underlying concepts:

# In[7]:


print(count.vocabulary_)

# As we can see from executing the preceding command, the vocabulary is stored in a Python dictionary, which maps the unique words that are mapped to integer indices. Next let us print the feature vectors that we just created:

# Each index position in the feature vectors shown here corresponds to the integer values that are stored as dictionary items in the CountVectorizer vocabulary. For example, the  rst feature at index position 0 resembles the count of the word and, which only occurs in the last document, and the word is at index position 1 (the 2nd feature in the document vectors) occurs in all three sentences. Those values in the feature vectors are also called the raw term frequencies: *tf (t,d)*—the number of times a term t occurs in a document *d*.

# In[8]:


print(bag.toarray())

# ## Assessing word relevancy via term frequency-inverse document frequency

# In[9]:


np.set_printoptions(precision=2)

# When we are analyzing text data, we often encounter words that occur across multiple documents from both classes. Those frequently occurring words typically don't contain useful or discriminatory information. In this subsection, we will learn about a useful technique called term frequency-inverse document frequency (tf-idf) that can be used to downweight those frequently occurring words in the feature vectors. The tf-idf can be de ned as the product of the term frequency and the inverse document frequency:
#
# $$\text{tf-idf}(t,d)=\text{tf (t,d)}\times \text{idf}(t,d)$$
#
# Here the tf(t, d) is the term frequency that we introduced in the previous section,
# and the inverse document frequency *idf(t, d)* can be calculated as:
#
# $$\text{idf}(t,d) = \text{log}\frac{n_d}{1+\text{df}(d, t)},$$
#
# where $n_d$ is the total number of documents, and *df(d, t)* is the number of documents *d* that contain the term *t*. Note that adding the constant 1 to the denominator is optional and serves the purpose of assigning a non-zero value to terms that occur in all training samples; the log is used to ensure that low document frequencies are not given too much weight.
#
# Scikit-learn implements yet another transformer, the `TfidfTransformer`, that takes the raw term frequencies from `CountVectorizer` as input and transforms them into tf-idfs:

# In[10]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs))
      .toarray())

# As we saw in the previous subsection, the word is had the largest term frequency in the 3rd document, being the most frequently occurring word. However, after transforming the same feature vector into tf-idfs, we see that the word is is
# now associated with a relatively small tf-idf (0.45) in document 3 since it is
# also contained in documents 1 and 2 and thus is unlikely to contain any useful, discriminatory information.
#

# However, if we'd manually calculated the tf-idfs of the individual terms in our feature vectors, we'd have noticed that the `TfidfTransformer` calculates the tf-idfs slightly differently compared to the standard textbook equations that we de ned earlier. The equations for the idf and tf-idf that were implemented in scikit-learn are:

# $$\text{idf} (t,d) = log\frac{1 + n_d}{1 + \text{df}(d, t)}$$
#
# The tf-idf equation that was implemented in scikit-learn is as follows:
#
# $$\text{tf-idf}(t,d) = \text{tf}(t,d) \times (\text{idf}(t,d)+1)$$
#
# While it is also more typical to normalize the raw term frequencies before calculating the tf-idfs, the `TfidfTransformer` normalizes the tf-idfs directly.
#
# By default (`norm='l2'`), scikit-learn's TfidfTransformer applies the L2-normalization, which returns a vector of length 1 by dividing an un-normalized feature vector *v* by its L2-norm:
#
# $$v_{\text{norm}} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v_{1}^{2} + v_{2}^{2} + \dots + v_{n}^{2}}} = \frac{v}{\big (\sum_{i=1}^{n} v_{i}^{2}\big)^\frac{1}{2}}$$
#
# To make sure that we understand how TfidfTransformer works, let us walk
# through an example and calculate the tf-idf of the word is in the 3rd document.
#
# The word is has a term frequency of 3 (tf = 3) in document 3, and the document frequency of this term is 3 since the term is occurs in all three documents (df = 3). Thus, we can calculate the idf as follows:
#
# $$\text{idf}("is", d3) = log \frac{1+3}{1+3} = 0$$
#
# Now in order to calculate the tf-idf, we simply need to add 1 to the inverse document frequency and multiply it by the term frequency:
#
# $$\text{tf-idf}("is",d3)= 3 \times (0+1) = 3$$

# In[11]:


tf_is = 3
n_docs = 3
idf_is = np.log((n_docs + 1) / (3 + 1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)

# If we repeated these calculations for all terms in the 3rd document, we'd obtain the following tf-idf vectors: [3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0 , 1.69, 1.29]. However, we notice that the values in this feature vector are different from the values that we obtained from the TfidfTransformer that we used previously. The  nal step that we are missing in this tf-idf calculation is the L2-normalization, which can be applied as follows:

# $$\text{tfi-df}_{norm} = \frac{[3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0 , 1.69, 1.29]}{\sqrt{[3.39^2, 3.0^2, 3.39^2, 1.29^2, 1.29^2, 1.29^2, 2.0^2 , 1.69^2, 1.29^2]}}$$
#
# $$=[0.5, 0.45, 0.5, 0.19, 0.19, 0.19, 0.3, 0.25, 0.19]$$
#
# $$\Rightarrow \text{tfi-df}_{norm}("is", d3) = 0.45$$

# As we can see, the results match the results returned by scikit-learn's `TfidfTransformer` (below). Since we now understand how tf-idfs are calculated, let us proceed to the next sections and apply those concepts to the movie review dataset.

# In[12]:


tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf

# In[13]:


l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf ** 2))
l2_tfidf

# ## Cleaning text data

# In[14]:


df.loc[0, 'review'][-50:]

# In[15]:


import re


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


# In[16]:


preprocessor(df.loc[0, 'review'][-50:])

# In[17]:


preprocessor("</a>This :) is :( a test :-)!")

# In[18]:


df['review'] = df['review'].apply(preprocessor)

# print(df)

# ## Processing documents into tokens

# In[9]:


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


import nltk

nltk.download('stopwords')

# In[13]:


from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
 if w not in stop]

# # Training a logistic regression model for document classification

# Strip HTML and punctuation to speed up the GridSearch later:

# In[24]:


data_clean = df.loc[:, ['review', 'sentiment']]
print(data_clean.head())

train, test = train_test_split(data_clean, test_size=0.2, random_state=1)
X_train = train['review'].values
X_test = test['review'].values
y_train = train['sentiment']
y_test = test['sentiment']

def tokenize(text):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)



en_stopwords = set(stopwords.words("english"))

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range=(1, 1),
    stop_words = en_stopwords)

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

np.random.seed(1)

pipeline_svm = make_pipeline(vectorizer,
                            SVC(probability=True, kernel="linear", class_weight="balanced"))

grid_svm = GridSearchCV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1]},
                    cv = kfolds,
                    scoring="roc_auc",
                    verbose=1,
                    n_jobs=-1)

grid_svm.fit(X_train, y_train)


print(grid_svm.score(X_test, y_test))

print(grid_svm.best_params_)
print(grid_svm.best_score_)


# from sklearn.externals import joblib
# joblib.dump(grid_svm.best_estimator_, 'test_trained.pkl')

def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result

report_results(grid_svm.best_estimator_, X_test, y_test)

def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr

roc_svm = get_roc_curve(grid_svm.best_estimator_, X_test, y_test)

fpr, tpr = roc_svm
plt.figure(figsize=(14,8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.show()

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = \
    learning_curve(grid_svm.best_estimator_, X_train, y_train, cv=5, n_jobs=-1,
                   scoring="roc_auc", train_sizes=np.linspace(.1, 1.0, 10), random_state=1)



def plot_learning_curve(X, y, train_sizes, train_scores, test_scores, title='', ylim=None, figsize=(14,8)):

    plt.figure(figsize=figsize)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="lower right")
    return plt

plot_learning_curve(X_train, y_train, train_sizes,
                    train_scores, test_scores, ylim=(0.7, 1.01), figsize=(14,6))
plt.show()


moby = "It might seem crazy what I'm 'bout to say Sunshine she's here, you can take a break I'm a hot air balloon that could go to space With the air, like I don't care baby by the way Huh, because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Here come bad news, talking this and that (Yeah) Well, give me all you got, and don't hold it back (Yeah) Well, I should probably warn you I'll be just fine (Yeah) No offense to you, don't waste your time Here's why Because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Hey, come on, uh Bring me down, can't nuthin' (happy) Bring me down My level is too high to bring me down (happy) Can't nuthin', bring me down (happy) I said, let me tell you now, unh (happy) Bring me down, can't nuthin', bring me down (happy, happy, happy) My level is too high to bring me down (happy, happy, happy) Can't nuthin' bring me down (happy, happy, happy) I said Because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Come on, unh bring me down can't nuthin' (happy, happy, happy) Bring me down my level is too high (happy, happy, happy) Bring me down can't nuthin' (happy, happy, happy) Bring me down, I said Because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you, eh eh eh Because I'm happy Clap along if you feel like that's what you wanna do Because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you, eh hey Because I'm happy Clap along if you feel like that's what you wanna do, heh come on"
imm = "I'm so tired of being here Suppressed by all my childish fears And if you have to leave I wish that you would just leave 'Cause your presence still lingers here And it won't leave me alone These wounds won't seem to heal, this pain is just too real There's just too much that time cannot erase When you cried, I'd wipe away all of your tears When you'd scream, I'd fight away all of your fears And I held your hand through all of these years But you still have all of me You used to captivate me by your resonating light Now, I'm bound by the life you left behind Your face it haunts my once pleasant dreams Your voice it chased away all the sanity in me These wounds won't seem to heal, this pain is just too real There's just too much that time cannot erase When you cried, I'd wipe away all of your tears When you'd scream, I'd fight away all of your fears And I held your hand through all of these years But you still have all of me I've tried so hard to tell myself that you're gone But though you're still with me, I've been alone all along When you cried, I'd wipe away all of your tears When you'd scream, I'd fight away all of your fears And I held your hand through all of these years You still have all of me, me, me"



print("moby")
print(grid_svm.predict([moby]))
print("immorta)")
print(grid_svm.predict([imm]))

# reference https://www.kaggle.com/lbronchal/sentiment-analysis-with-svm
