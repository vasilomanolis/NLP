import pandas as pd
from PyLyrics import *
import lyricscorpora
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import time

style.use('ggplot')

songsLibraryOfEmotion = {'SongIndex': [0, 0, 0, 0, 0, 0, "Title", "Artist"],
                         'ML1': [0, 0, 0, 0, 0, 0, "Title", "Artist"], }

df2 = pd.DataFrame(songsLibraryOfEmotion)

#
# print(web_stats)
# print(df)
# print(df.set_index('Day'))


df = pd.read_csv('/Users/Vasilis/Desktop/Lennon/Ratings_Warriner_et_al.csv')



# could add pickle

df2 = df.set_index("Word", drop=False)


def remove_stop_words(lyrics):
    stop_words = set(stopwords.words('english'))
    lyrics = lyrics.lower()
    word_tokens = word_tokenize(lyrics)

    filtered_lyrics = [w for w in word_tokens if not w in stop_words]

    return filtered_lyrics


# Adds a song in the dictionary in the dictionary and it initializes
# its emotion state as zero.
def initialize_emotion_state_of_song(songIndex):
    songsLibraryOfEmotion[songIndex] = [0, 0, 0, 0, 0, 0]

    song_avg_valence = 0
    song_avg_arousal = 0
    song_avg_dominance = 0
    song_sd_valence = 0
    song_sd_arousal = 0
    song_sd_dominance = 0


# Calculates the emotion state of a song
def get_song_emotion(songIndex, lyrics):
    # if song_title is new call the initialize else continue

    # for each word of lyrics
    for word in remove_stop_words(lyrics):

        # for word in lyrics.split():
        # if not word.isspace():
        # print(word)
        # removes punctuation
        # word = re.sub(r'[^\w\s]', '', word)

        #maybe first lemmatize?
        word_emotion = get_emotion_for_word(word)
        for i in range(7):
            print(songsLibraryOfEmotion[songIndex][i])
            print(songsLibraryOfEmotion[songIndex][i])
            print(word_emotion[i])

            songsLibraryOfEmotion[songIndex][i] = songsLibraryOfEmotion[songIndex][i] + word_emotion[i]
            print("i'm printing" + songsLibraryOfEmotion[songIndex])

    return 1
    #return songsLibraryOfEmotion[songIndex]

# It
def get_emotion_for_word(word):
    if word in df2.index:
        word_info = df2.loc[word, :]

        print("the word info is", word_info)

        avg_valence = word_info[2]
        avg_arousal = word_info[5]
        avg_dominance = word_info[8]
        sd_valence = word_info[3]
        sd_arousal = word_info[6]
        sd_dominance = word_info[9]
        unknown = 0

        print("\nThe word ", word, " was found in Lexicon.")
        print("avg valence: ", avg_valence, "avg arousal: ", avg_arousal, "avg dominance: ", avg_dominance, "vsd_alence: ", sd_valence, "sd_arousal: ", sd_arousal, "sd_dominance: ", sd_dominance, unknown, ".\n")


    #

    else:
        print("The word ", word, " was NOT found in Lexicon")
        avg_valence = 0
        avg_arousal = 0
        avg_dominance = 0
        sd_valence = 0
        sd_arousal = 0
        sd_dominance = 0
        unknown = 1

    wordSentiment = (avg_valence, avg_arousal, avg_dominance, sd_valence, sd_arousal, sd_dominance, unknown)

    return wordSentiment


#moby = "Hush, hush Don't say a word The faint cries can hardly be heard A storm lies beyond the horizon, barely Don’t stop Sweep through the days Like children that can’t stay awake Stay here untainted and say Stay while the melody’s sung Break like a wave on the run I do be sure I can’t say anymore I just know that it won’t last forever Rush, rush Take me away Like hourglass sand that never escapes Stars are born and then die, but carefree A small clock that ticks without time And watched by an ocean of eyes Ending, ascending and then Stay while the melody’s sung Break like a wave on the run I do be sure I can’t say anymore I just know that it won’t last I just know that it won’t last forever"

moby = 'abalone abandon'

# test_song_id = 'ML1'
# test7 = lyrics_database.iloc['ML1'][3]


# results = (get_song_emotion(test_song_id, lyrics))


path = '/Users/Vasilis/Desktop/Lennon/MoodyLyrics/test5.csv'


def tag_database(path):
    lyrics_database = pd.read_csv(path)
    lyrics_database = lyrics_database.set_index("Index", drop=False)
    taggedSongs = {}

    for index, row in lyrics_database.iterrows():
        # print (index)
        # print (row)
        # print (row["Index"])
        songIndex = row["Index"]
        songTitle = row["Title"]
        songArtist = row["Artist"]
        songLyrics = row["Lyrics"]
        songsLibraryOfEmotion[songIndex] = [0, 0, 0, 0, 0, 0, 0,  songTitle, songArtist]
        taggedSongs[songIndex] = [0, 0, 0, 0, 0, 0, songTitle, songArtist]
        # print(taggedSongs)
        get_song_emotion(songIndex, songLyrics)
    return taggedSongs


# tag_database(path)
#print(tag_database(path))

# print("\nThe song has is ", "positive: ", results[0], ", negative: ", results[1], ", anger: ", results[2],
#       ", anticipation: ", results[3], ", disgust: ",
#       results[4], ", fear: ", results[5], ", joy: ", results[6], ", sadness: ", results[7], ", surprise: ", results[8],
#       ", trust: ", results[9],
#       ", unknown: ", results[10])

# print(lyrics)

# print(word_info[105:115])
# print(positive)
# print(negative)
# print(anger)

#print(get_emotion_for_word('abandon'))
print(get_song_emotion('ML1', moby))
