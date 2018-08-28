# Author: VM
# A class to download the lyrics of the corpora.

import pandas as pd
from PyLyrics import *

import csv

lyrics_doc1 = pd.read_csv('/Users/Vasilis/Desktop/Lennon/MoodyLyrics/ml_pn_balanced.csv')
lyrics_doc2 = pd.read_csv('/Users/Vasilis/Desktop/Lennon/MoodyLyrics/ml_balanced.csv')
lyrics_doc1 = lyrics_doc1.set_index('Index', drop=False)
lyrics_doc2 = lyrics_doc2.set_index('Index', drop=False)

lyrics_doc1['Lyrics'] = ''

lyrics_doc1.set_index("Index", drop=True, inplace=True)
LyricsLibraryTwoDimensions = lyrics_doc1.to_dict(orient="index")

print(lyrics_doc1)
print(LyricsLibraryTwoDimensions['ML1']['Artist'], LyricsLibraryTwoDimensions['ML1']['Title'])
# print(PyLyrics.getLyrics(LyricsLibraryTwoDimensions['ML1']['Artist'], LyricsLibraryTwoDimensions['ML1']['Title']))

for Index in LyricsLibraryTwoDimensions:
    artist = LyricsLibraryTwoDimensions[Index]['Artist']
    title = LyricsLibraryTwoDimensions[Index]['Title']

    try:
        lyrics = PyLyrics.getLyrics(artist, title)
    except:
        lyrics = "Lyrics unavailable"

    LyricsLibraryTwoDimensions[Index]['Lyrics'] = lyrics
    # print(LyricsLibraryTwoDimensions[Index]['Lyrics'])

print(LyricsLibraryTwoDimensions)

# (pd.DataFrame.from_dict(data=LyricsLibraryTwoDimensions, orient='index')
#   .to_csv('dict_file.csv', header=True))

# pd.DataFrame.from_dict(data=LyricsLibraryTwoDimensions, orient='index', columns=['index', 'artist', 'title', 'pos/neg', 'lyrics'])

# print(PyLyrics.getLyrics('The Beatles','Yellow submarine'))
