

# ## Topic modeling

# ### Decomposing text documents with Latent Dirichlet Allocation

# ### Latent Dirichlet Allocation with scikit-learn

# In[1]:


import pandas as pd

df = pd.read_csv('genre_finder_test2.csv', encoding='utf-8', low_memory=False)
print(df.head(3))
print(df)

df = df.head(260000)


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['review'].values)

# In[3]:


from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=14,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)

# In[4]:


lda.components_.shape

# In[5]:


n_top_words = 15
feature_names = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort() \
                        [:-n_top_words - 1:-1]]))

# Based on reading the 5 most important words for each topic, we may guess that the LDA identified the following topics:
#
# 1. Generally bad movies (not really a topic category)
# 2. Movies about families
# 3. War movies
# 4. Art movies
# 5. Crime movies
# 6. Horror movies
# 7. Comedies
# 8. Movies somehow related to TV shows
# 9. Movies based on books
# 10. Action movies

# To confirm that the categories make sense based on the reviews, let's plot 5 movies from the horror movie category (category 6 at index position 5):

# In[6]:
category0 = X_topics[:, 0].argsort()[::-1]

for iter_idx, movie_idx in enumerate(category0[:10]):
    print('\nGenre category #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:10000], '...')

category1 = X_topics[:, 1].argsort()[::-1]

for iter_idx, movie_idx in enumerate(category1[:10]):
    print('\nGenre category #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:10000], '...')

category2 = X_topics[:, 2].argsort()[::-1]

for iter_idx, movie_idx in enumerate(category2[:10]):
    print('\nGenre category #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:10000], '...')


category3 = X_topics[:, 3].argsort()[::-1]

for iter_idx, movie_idx in enumerate(category3[:10]):
    print('\nGenre category #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:10000], '...')

category4 = X_topics[:, 4].argsort()[::-1]

for iter_idx, movie_idx in enumerate(category4[:10]):
    print('\nGenre category #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:10000], '...')


# category5 = X_topics[:, 5].argsort()[::-1]
#
# for iter_idx, movie_idx in enumerate(category5[:10]):
#     print('\nGenre category #%d:' % (iter_idx + 1))
#     print(df['review'][movie_idx][:10000], '...')
#
#
# category6 = X_topics[:, 6].argsort()[::-1]
#
# for iter_idx, movie_idx in enumerate(category6[:10]):
#     print('\nGenre category #%d:' % (iter_idx + 1))
#     print(df['review'][movie_idx][:10000], '...')
#
# category7 = X_topics[:, 7].argsort()[::-1]
#
# for iter_idx, movie_idx in enumerate(category7[:10]):
#     print('\nGenre category #%d:' % (iter_idx + 1))
#     print(df['review'][movie_idx][:10000], '...')
#
# category8 = X_topics[:, 8].argsort()[::-1]
#
#
# for iter_idx, movie_idx in enumerate(category8[:10]):
#     print('\nGenre category #%d:' % (iter_idx + 1))
#     print(df['review'][movie_idx][:10000], '...')
#
# category9 = X_topics[:, 9].argsort()[::-1]
#
#
# for iter_idx, movie_idx in enumerate(category9[:10]):
#     print('\nGenre category #%d:' % (iter_idx + 1))
#     print(df['review'][movie_idx][:10000], '...')
#
# category10 = X_topics[:, 10].argsort()[::-1]
#
#
# for iter_idx, movie_idx in enumerate(category10[:10]):
#     print('\nGenre category #%d:' % (iter_idx + 1))
#     print(df['review'][movie_idx][:10000], '...')
#
# category11 = X_topics[:, 11].argsort()[::-1]
#
#
# for iter_idx, movie_idx in enumerate(category11[:10]):
#     print('\nGenre category #%d:' % (iter_idx + 1))
#     print(df['review'][movie_idx][:10000], '...')
#
# category12 = X_topics[:, 12].argsort()[::-1]
#
#
# for iter_idx, movie_idx in enumerate(category12[:10]):
#     print('\nGenre category #%d:' % (iter_idx + 1))
#     print(df['review'][movie_idx][:10000], '...')
#
# category13 = X_topics[:, 13].argsort()[::-1]
#
#
# for iter_idx, movie_idx in enumerate(category13[:10]):
#     print('\nGenre category #%d:' % (iter_idx + 1))
#     print(df['review'][movie_idx][:10000], '...')
#
# category14 = X_topics[:, 14].argsort()[::-1]
#
#
# for iter_idx, movie_idx in enumerate(category14[:10]):
#     print('\nGenre category #%d:' % (iter_idx + 1))
#     print(df['review'][movie_idx][:10000], '...')


