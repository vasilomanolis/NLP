# Author: VM
# An implementation of emotion analysis using the Warriner lexicon.
# The implementation is based on calculating the Valence, Arousal, and Dominance of each song.

import pandas as pd
from matplotlib import style
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

style.use('ggplot')

# A dictionary that will be used to build a library of songs lyrics and their emotions.

songs_library_of_emotion = {'SongIndex': ["Title", "Artist",
                                          "valence", "arousal", "dominance", "empty", "empty", "empty", "empty",
                                          "empty", "empty", "empty", "unknown",
                                          "total_number_of_words", "emotional_impact_words", "neutral_words",
                                          "norm_valence", "norm_arousal", "norm_dominance", "empty",
                                          "empty", "empty", "empty",
                                          "empty", "empty", "empty", "norm_unknown",
                                          "norm_total_number_of_words", "norm_emotional_impact_words",
                                          "norm_neutral_words", ],
                            'ML1': ["Title", "Artist", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0],
                            'ML2': ["Title", "Artist", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0],
                            }

# We read the Warriner Lexicon with a help of a dataframe.

df = pd.read_csv('/Users/Vasilis/Desktop/Lennon/Ratings_Warriner_et_al.csv')

# We select index.

df2 = df.set_index("Word", drop=False)


# This methods accepts a word and returns a list with its sentiment.
# Condition 1: If the word is included in the Warriner lexicon then we fetch the word sentiment.
# Condition 2: If the word is NOT included in the Warriner lexicon, we lemmatize the word and try again.
# Condition 3: If the word and its lemma is NOT included in the Warriner lexicon, the emotion values are set to zero.
def get_emotion_for_word(word):
    lemmatizer = WordNetLemmatizer()
    lemma_word = lemmatizer.lemmatize(word)
    empty = 0

    # Condition 1: If the word is included in the Warriner lexicon then we fetch the word sentiment.

    if word in df2.index:
        word_info = df2.loc[word, :]

        avg_valence = word_info[2]
        avg_arousal = word_info[5]
        avg_dominance = word_info[8]
        sd_valence = word_info[3]
        sd_arousal = word_info[6]
        sd_dominance = word_info[9]
        unknown = 0

        # print("\nThe word ", word, " was found in Lexicon.")
        # print("avg valence: ", avg_valence, "avg arousal: ", avg_arousal, "avg dominance: ", avg_dominance,
        #       "vsd_alence: ", sd_valence, "sd_arousal: ", sd_arousal, "sd_dominance: ", sd_dominance, "unknown words", unknown, ".\n")

    # Condition 2: If the word is NOT included in the Warriner lexicon, we lemmatize the word and try again.
    elif lemma_word in df2.index:

        word_info = df2.loc[lemma_word, :]

        avg_valence = word_info[2]
        avg_arousal = word_info[5]
        avg_dominance = word_info[8]
        sd_valence = word_info[3]
        sd_arousal = word_info[6]
        sd_dominance = word_info[9]
        unknown = 0

        # print("\nThe word ", word, " was NOT found in Lexicon but the lemmatized word ", lemma_word, " was found.")
        # print("avg valence: ", avg_valence, "avg arousal: ", avg_arousal, "avg dominance: ", avg_dominance,
        #       "vsd_alence: ", sd_valence, "sd_arousal: ", sd_arousal, "sd_dominance: ", sd_dominance, "unknown words", unknown, ".\n")

    # Condition 3: If the word and its lemma is NOT included in the Warriner lexicon, the emotion values are set to zero.
    else:
        # print("The word ", word, " was NOT found in Lexicon")
        avg_valence = 0
        avg_arousal = 0
        avg_dominance = 0
        sd_valence = 0
        sd_arousal = 0
        sd_dominance = 0
        unknown = 1

    # Return the sentiment of the word in the following format
    # We have some empty slots for scaling the system in the future.
    word_sentiment = (
        avg_valence, avg_arousal, avg_dominance, sd_valence, sd_arousal, sd_dominance, empty, empty, empty, empty,
        unknown)
    return word_sentiment


# Accepts a song_index, it adds the song in a dictionary and initializes its emotion state to zero.
def initialize_emotion_state_of_song(song_index):
    # The zeros are represeting the following values resepctively:
    # [positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, trust, unknown].
    songs_library_of_emotion[song_index] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0]


# A method that accepts the lyrics of a song and returns the lyrics after
# forcing lower case, removing the English stop words and punctuation.
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


# This method accepts a songIndex and the lyrics of a song and returns its emotion state.
# PRE-CONDITIONS: The SongIndex should have been already added in the songs_library_of_emotion.
def get_song_emotion(song_index, lyrics):

    try:
        filtered_lyrics = clean_lyrics(lyrics)
    except:
        filtered_lyrics = lyrics

    # print(filtered_lyrics)
    total_number_of_words = len(filtered_lyrics)
    # print(total_number_of_words)

    for word in filtered_lyrics:
        # for word in lyrics.split():
        # if not word.isspace():
        # print(word)
        # removes punctuation
        # word = re.sub(r'[^\w\s]', '', word)

        # kane lemma to word

        # Calculating the Arousal, Valence, and Dominance values of a song.

        word_emotion = get_emotion_for_word(word)
        for i in range(2, 13):
            # print("now i print songs_library_of_emotion[song_index][i]", songs_library_of_emotion[song_index][i])
            # print("now i print word_emotion[i - 2]",  word_emotion[i - 2])
            # print(word)
            songs_library_of_emotion[song_index][i] = songs_library_of_emotion[song_index][i] + word_emotion[i - 2]


    # The position 11 (12th position, counting form 0) is the number of words.

    song_title = songs_library_of_emotion[song_index][0]
    artist = songs_library_of_emotion[song_index][1]

    unknown = songs_library_of_emotion[song_index][12]
    emotional_impact_words = total_number_of_words - unknown

    # print("unknown words are ", unknown)
    emotional_impact_words = total_number_of_words - unknown

    for i in range(2, 8):
        # print("now i print songs_library_of_emotion[song_index][i]", songs_library_of_emotion[song_index][i])
        # print("now i print word_emotion[i - 2]",  word_emotion[i - 2])
            songs_library_of_emotion[song_index][i] = avg_emotion_value((songs_library_of_emotion[song_index][i]), emotional_impact_words)


    valence = songs_library_of_emotion[song_index][2]
    arousal = songs_library_of_emotion[song_index][3]
    dominance = songs_library_of_emotion[song_index][4]

    neutral_words = 0

    songs_library_of_emotion[song_index][13] = total_number_of_words
    songs_library_of_emotion[song_index][14] = emotional_impact_words
    songs_library_of_emotion[song_index][15] = 0

    songs_library_of_emotion[song_index][16] = norm_valence = normalize_emotion_value(valence, total_number_of_words,
                                                                                      unknown)
    songs_library_of_emotion[song_index][17] = norm_arousal = normalize_emotion_value(arousal, total_number_of_words,
                                                                                      unknown)
    songs_library_of_emotion[song_index][18] = norm_dominance = normalize_emotion_value(dominance,
                                                                                        total_number_of_words,
                                                                                        unknown)

    songs_library_of_emotion[song_index][26] = norm_unknown = avg_emotion_value(unknown, total_number_of_words) * 100
    songs_library_of_emotion[song_index][27] = norm_total_number_of_words = 100
    songs_library_of_emotion[song_index][28] = norm_emotional_impact_words = avg_emotion_value(
                                                                                     emotional_impact_words, total_number_of_words) * 100
    songs_library_of_emotion[song_index][29] = norm_neutral_words = avg_emotion_value(neutral_words , total_number_of_words) * 100

    # print("The song ", song_title, " (", artist, ", ", song_index, ") has: ")
    # print("Valence: ", valence, "or normalized %", norm_valence)
    # print("Arousal: ", arousal, "or normalized %", norm_arousal)
    # print("Dominance: ", dominance, "or normalized %", norm_dominance)
    # print("Unknown words: ", unknown, "or normalized %", norm_unknown)
    # print("Total Number of Words: ", total_number_of_words, "or normalized %", norm_total_number_of_words)
    # print("Words with emotional impact: ", emotional_impact_words, "or normalized %", norm_emotional_impact_words)
    # print("Empty slot: ", neutral_words, "or normalized %", norm_neutral_words)

    return songs_library_of_emotion[song_index]


# A method to normalize the absolute values in order to make them comparable with each other. [min = 1, max = 9]
def normalize_emotion_value(emotion_value, total_number_of_words, unknown_words):
    if total_number_of_words == 0:
        normalized_emotion_value = 0
    else:
        normalized_emotion_value = (emotion_value * 100) / (9)

    return normalized_emotion_value


def avg_emotion_value(emotion_value, emotional_impact_words):
    if emotional_impact_words == 0:
        normalized_emotion_value = 0
    else:
        normalized_emotion_value = (emotion_value) / (emotional_impact_words)

    return normalized_emotion_value


# A method that accepts a path to a database of lyrics in a CSV format and creates a CSV files with the results:
# PRE-CONDITIONS: The database should be in CSV format. It should have four columns: [index], [Title], [Artist], [lyrics]
# POST-CONDITIONS: A CSV file with the results has been created.
def tag_database(file_path, output_name):
    lyrics_database = pd.read_csv(file_path, encoding='latin-1')

    lyrics_database = lyrics_database.set_index("Index", drop=False)
    tagged_songs = {}

    for index, row in lyrics_database.iterrows():
        song_index = row["Index"]
        song_title = row["Title"]
        song_artist = row["Artist"]
        song_lyrics = row["Lyrics"]
        songs_library_of_emotion[song_index] = [song_title, song_artist, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0]
        tagged_songs[song_index] = [song_title, song_artist, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0,
                                    0, 0, 0, 0, 0, 0]

        # print(tagged_songs)
        get_song_emotion(song_index, song_lyrics)

        the_header = ["title", "artist", "valence", "arousal", "dominance", "sd_valence", "sd_arousal",
                      "sd_dominance", "empty", "empty", "empty", "empty",
                      "unknown", "total_number_of_words", "emotional_impact_words",
                      "empty", "norm_valence", "norm_arousal", "norm_dominance",
                      "empty", "empty", "empty",
                      "empty", "empty", "empty", "empty",
                      "norm_unknown", "norm_total", "norm_emotional_impact", "empty"]

        for i in range(2, 30):
            tagged_songs[song_index][i] = songs_library_of_emotion[song_index][i]

    (pd.DataFrame.from_dict(data=tagged_songs, orient='index')
     .to_csv(output_name, header=the_header))
    return tagged_songs


# A alternative method to include information about the genre and the year of release.
# It accepts a path to a database of lyrics in a CSV format and creates a CSV files with the results:
# PRE-CONDITIONS: The database should be in CSV format. It should have four columns: [index], [Title], [Artist], [lyrics], [genre], [year]
# POST-CONDITIONS: A CSV file with the results has been created.
def tag_big_lyrics(file_path, output_name):
    # lyrics_database = pd.read_csv(file_path)
    lyrics_database = pd.read_csv(file_path, encoding='latin-1', na_filter=False)
    lyrics_database = lyrics_database.set_index("index", drop=False)
    tagged_songs = {}

    for index, row in lyrics_database.iterrows():
        # print (index)
        # print (row)
        # print (row["Index"])
        song_index = row["index"]
        song_title = row["song"]
        song_artist = row["artist"]
        song_lyrics = row["lyrics"]
        song_genre = row["genre"]
        song_release_year = row["year"]
        songs_library_of_emotion[song_index] = [song_title, song_artist, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, song_genre, song_release_year]
        tagged_songs[song_index] = [song_title, song_artist, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0,
                                    0, 0, 0, 0, 0, 0, song_genre, song_release_year]

        # print(tagged_songs)
        get_song_emotion(song_index, song_lyrics)

        the_header = ["title", "artist", "valence", "arousal", "dominance", "sd_valence", "sd_arousal",
                      "sd_dominance", "empty", "empty", "empty", "empty",
                      "unknown", "total_number_of_words", "emotional_impact_words",
                      "empty", "norm_valence", "norm_arousal", "norm_dominance",
                      "empty", "empty", "empty",
                      "empty", "empty", "empty", "empty",
                      "norm_unknown", "norm_total", "norm_emotional_impact", "empty", "genre", "year"]

        for i in range(2, 30):
            tagged_songs[song_index][i] = songs_library_of_emotion[song_index][i]

    (pd.DataFrame.from_dict(data=tagged_songs, orient='index')
     .to_csv(output_name, header=the_header))
    return tagged_songs


# Some songs for testing

dontletmedown = "Don't let me down, don't let me down Don't let me down, don't let me down Nobody ever loved me like she does Oh, she does, yeah, she does And if somebody loved me like she do me Oh, she do me, yes, she does Don't let me down, don't let me down Don't let me down, don't let me down I'm in love for the first time Don't you know it's gonna last It's a love that lasts forever It's a love that had no past Don't let me down, don't let me down Don't let me down, don't let me down And from the first time that she really done me Oh, she done me, she done me good I guess nobody ever really done me Oh, she done me, she done me good Don't let me down, hey, don't let me down Heee! Don't let me down Don't let me down Don't let me down, don't let me let down Can you dig it? Don't let me down"

sep = "happy happy sad"
yellow = "In the town where I was born Lived a man who sailed to sea And he told us of his life In the land of submarines So we sailed up to the sun Till we found a sea of green And we lived beneath the waves In our yellow submarine We all live in a yellow submarine Yellow submarine, yellow submarine We all live in a yellow submarine Yellow submarine, yellow submarine And our friends are all aboard Many more of them live next door And the band begins to play We all live in a yellow submarine Yellow submarine, yellow submarine We all live in a yellow submarine Yellow submarine, yellow submarine (Full speed ahead Mr. Boatswain, full speed ahead Full speed ahead it is, Sergeant. Cut the cable, drop the cable Aye, Sir, aye Captain, captain) As we live a life of ease Every one of us has all we need Sky of blue and sea of green In our yellow submarine We all live in a yellow submarine Yellow submarine, yellow submarine We all live in a yellow submarine Yellow submarine, yellow submarine We all live in a yellow submarine Yellow submarine, yellow submarine"

# print(get_song_emotion('ML1', sep))
# print(get_song_emotion('ML2', dontletmedown))

# Perform song tagging.

# Tag the MoodyLyrics database

# Source: https://dl.acm.org/citation.cfm?id=3059340
# path_ml_pn_balanced_with_lyrics = "/Users/vasilis/Desktop/Lennon/Databases/MoodyLyrics/ml_pn_balanced_with_lyrics.csv"
# output_ml_pn_balanced_with_lyrics = "tagged_ml_pn_balanced_with_lyrics_Warriner.csv"
# tag_database(path_ml_pn_balanced_with_lyrics, output_ml_pn_balanced_with_lyrics)


# Tag the Top Billboard database
# path_top_billboard = "/Users/vasilis/Desktop/Lennon/Databases/billboard_lyrics_1964-2015.csv"
# output_top_billboard = "tagged_top_billboard_Warriner.csv"
# tag_big_lyrics(path_top_billboard, output_top_billboard)

# Tag song database with 300.000 songs' lyrics
# Source: https://www.kaggle.com/karineh/40-year-lyrics-evolution/data?scriptVersionId=1930166
path_top_billboard = "/Users/vasilis/Desktop/Lennon/Databases/big_lyrics.csv"
output_top_billboard = "tagged_big_lyrics_Warriner.csv"
tag_big_lyrics(path_top_billboard, output_top_billboard)
