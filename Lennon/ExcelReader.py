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



style.use('ggplot')

songsLibraryOfEmotion = {'Imagine': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'Moby': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], }

df2 = pd.DataFrame(songsLibraryOfEmotion)

#
# print(web_stats)
# print(df)
# print(df.set_index('Day'))


df = pd.read_csv('/Users/Vasilis/Desktop/Lennon/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.csv')

el = "Greek (el)"
en = "English (en)"
de = 'German (de)'
fr = 'French (fr)'


df2 = df.set_index(en, drop=False)


def remove_stop_words(lyrics):
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(lyrics)

    filtered_lyrics = [w for w in word_tokens if not w in stop_words]

    return filtered_lyrics




# Adds a song in the dictionary in the dictionary and it initializes
# its emotion state as zero.
def initialize_emotion_state_of_song(song_title):
    songsLibraryOfEmotion[song_title] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    song_positive = 0
    song_negative = 0
    song_anger = 0
    song_anticipation = 0
    song_disgust = 0
    song_fear = 0
    song_joy = 0
    song_sadness = 0
    song_surprise = 0
    song_trust = 0
    song_unknown = 0


# Calculates the emotion state of a song
def get_song_emotion(song_title, lyrics):
    # if song_title is new call the initialize else continue


    # for each word of lyrics
    for word in remove_stop_words(lyrics):
    #for word in lyrics.split():
        # if not word.isspace():
        # print(word)
        # removes punctuation
        #word = re.sub(r'[^\w\s]', '', word)
        word_emotion = get_emotion_for_word(word)
        for i in range(11):
            songsLibraryOfEmotion[song_title][i] = songsLibraryOfEmotion[song_title][i] + word_emotion[i]

    return songsLibraryOfEmotion[song_title]


# It
def get_emotion_for_word(word):
    if word in df2.index:
        word_info = df2.loc[word, :]

        #print("the word info is", word_info)


        positive = word_info[105]
        negative = word_info[106]
        anger = word_info[107]
        anticipation = word_info[108]
        disgust = word_info[109]
        fear = word_info[110]
        joy = word_info[111]
        sadness = word_info[112]
        surprise = word_info[113]
        trust = word_info[114]
        unknown = 0

        print("\nThe word ", word, " was found in Lexicon.")
        print("positive: ", positive, "negative: ", negative, "anger: ", anger, "anticipation: ", anticipation,
              "disgust: ", disgust, "fear: ", fear, "joy: ", joy, "sadness: ", sadness, "surprise: ", surprise,
              "trust: ", trust, "unknown: ", unknown, ".\n")

    #

    else:
        print("The word ", word, " was NOT found in Lexicon")
        positive = 0
        negative = 0
        anger = 0
        anticipation = 0
        disgust = 0
        fear = 0
        joy = 0
        sadness = 0
        surprise = 0
        trust = 0
        unknown = 1

    wordSentiment = (positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, trust, unknown)

    return wordSentiment


imagine = "Imagine there's no heaven It's easy if you try No hell below us " \
          "Above us only sky Imagine all the people living for today Imagine there's " \
          "no countries It isn't hard to do Nothing to kill or die for And no religion too " \
          "Imagine all the people living life in peace, you You may say I'm a dreamer But I'm " \
          "not the only one I hope some day you'll join us And the world will be as one Imagine " \
          "no possessions I wonder if you can No need for greed or hunger A brotherhood of man " \
          "Imagine all the people sharing all the world, you You may say I'm a dreamer But I'm " \
          "not the only one I hope some day you'll join us And the world will be as one"





moby = "Love ourselves, and love"

results = (get_song_emotion('Moby', moby))

print("\nThe song has is ", "positive: ", results[0], ", negative: ", results[1], ", anger: ", results[2],
      ", anticipation: ", results[3], ", disgust: ",
      results[4], ", fear: ", results[5], ", joy: ", results[6], ", sadness: ", results[7], ", surprise: ", results[8],
      ", trust: ", results[9],
      ", unknown: ", results[10])


print(PyLyrics.getLyrics('Evanescence','Say you will')) #Print the lyrics directly



# print(word_info[105:115])
# print(positive)
# print(negative)
# print(anger)



