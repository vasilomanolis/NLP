import pandas as pd
from matplotlib import style
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

style.use('ggplot')

# A dictionary
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

# df2 = pd.DataFrame(songsLibraryOfEmotion)


df = pd.read_csv('/Users/Vasilis/Desktop/Lennon/Ratings_Warriner_et_al.csv')


# We create a data frame of the NRC lexicon, selecting the  column "Word" as index.
df2 = df.set_index("Word", drop=False)


# This methods accepts a word and returns a list with its sentiment.
# Condition 1: If the word is included in the NRC lexicon then we fetch the word sentiment.
# Condition 2: If the word is NOT included in the NRC lexicon, the emotion values are set to zero.
def get_emotion_for_word(word):
    lemmatizer = WordNetLemmatizer()
    lemma_word = lemmatizer.lemmatize(word)
    empty = 0
    # print("word and lemma respectively: ", word, " ", lemma_word)

    # If the word is included in the Lexicon.
    if word in df2.index:
        word_info = df2.loc[word, :]

        avg_valence = word_info[2]
        avg_arousal = word_info[5]
        avg_dominance = word_info[8]
        sd_valence = word_info[3]
        sd_arousal = word_info[6]
        sd_dominance = word_info[9]
        unknown = 0



        print("\nThe word ", word, " was found in Lexicon.")
        print("avg valence: ", avg_valence, "avg arousal: ", avg_arousal, "avg dominance: ", avg_dominance,
              "vsd_alence: ", sd_valence, "sd_arousal: ", sd_arousal, "sd_dominance: ", sd_dominance, "unknown words", unknown, ".\n")


    elif lemma_word in df2.index:

        word_info = df2.loc[lemma_word, :]

        avg_valence = word_info[2]
        avg_arousal = word_info[5]
        avg_dominance = word_info[8]
        sd_valence = word_info[3]
        sd_arousal = word_info[6]
        sd_dominance = word_info[9]
        unknown = 0


        print("\nThe word ", word, " was NOT found in Lexicon but the lemmatized word ", lemma_word, " was found.")
        print("avg valence: ", avg_valence, "avg arousal: ", avg_arousal, "avg dominance: ", avg_dominance,
              "vsd_alence: ", sd_valence, "sd_arousal: ", sd_arousal, "sd_dominance: ", sd_dominance, "unknown words", unknown, ".\n")

    # Else, if the word is NOT included in the Lexicon.
    else:
        print("The word ", word, " was NOT found in Lexicon")
        avg_valence = 0
        avg_arousal = 0
        avg_dominance = 0
        sd_valence = 0
        sd_arousal = 0
        sd_dominance =0
        unknown = 1

    # Return the sentiment of the word in the following format
    word_sentiment = (avg_valence, avg_arousal, avg_dominance, sd_valence, sd_arousal, sd_dominance, empty, empty, empty, empty, unknown)



    return word_sentiment


# Accepts a song_index, it adds the son in a dictionary and initialized its emotion state to zero.
def initialize_emotion_state_of_song(song_index):
    # The zeros are represeting the following values resepctively:
    # [positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, trust, unknown].
    songs_library_of_emotion[song_index] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0]


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


# This method accepts a songIndex and the lyrics of a song and returns its emotion state.
# PRE-CONDITIONS: The songIndex has to
def get_song_emotion(song_index, lyrics):
    # if song_title is new call the initialize else continue

    # for each word of lyrics

    # filter lyrics
    try:
        filtered_lyrics = clean_lyrics(lyrics)
    except:
        filtered_lyrics = lyrics

    #print(filtered_lyrics)
    total_number_of_words = len(filtered_lyrics)
    #print(total_number_of_words)

    for word in filtered_lyrics:
        # for word in lyrics.split():
        # if not word.isspace():
        # print(word)
        # removes punctuation
        # word = re.sub(r'[^\w\s]', '', word)

        # kane lemma to word

        word_emotion = get_emotion_for_word(word)
        for i in range(2, 13):
            # print("now i print songs_library_of_emotion[song_index][i]", songs_library_of_emotion[song_index][i])
            # print("now i print word_emotion[i - 2]",  word_emotion[i - 2])



            songs_library_of_emotion[song_index][i] = songs_library_of_emotion[song_index][i] + word_emotion[i - 2]

    # The position 11 (12th position, counting form 0) is the number of words.

    song_title = songs_library_of_emotion[song_index][0]
    artist = songs_library_of_emotion[song_index][1]

    unknown = songs_library_of_emotion[song_index][12]
    emotional_impact_words = total_number_of_words - unknown

    print("unknown words are ", unknown)
    emotional_impact_words = total_number_of_words - unknown

    for i in range(2, 8):
        # print("now i print songs_library_of_emotion[song_index][i]", songs_library_of_emotion[song_index][i])
        # print("now i print word_emotion[i - 2]",  word_emotion[i - 2])

        songs_library_of_emotion[song_index][i] = avg_emotion_value((songs_library_of_emotion[song_index][i] + word_emotion[i - 2]), emotional_impact_words)

    valence = songs_library_of_emotion[song_index][2]
    arousal =  songs_library_of_emotion[song_index][3]
    dominance = songs_library_of_emotion[song_index][4]



    neutral_words = total_number_of_words - (emotional_impact_words + unknown)

    songs_library_of_emotion[song_index][13] = total_number_of_words
    songs_library_of_emotion[song_index][14] = emotional_impact_words
    songs_library_of_emotion[song_index][15] = neutral_words

    songs_library_of_emotion[song_index][16] = norm_valence = normalize_emotion_value(valence, total_number_of_words,
                                                                                       unknown)
    songs_library_of_emotion[song_index][17] = norm_arousal = normalize_emotion_value(arousal, total_number_of_words,
                                                                                       unknown)
    songs_library_of_emotion[song_index][18] = norm_dominance = normalize_emotion_value(dominance, total_number_of_words,
                                                                                    unknown)
    # songs_library_of_emotion[song_index][19] = norm_anticipation = normalize_emotion_value(anticipation,
    #                                                                                        total_number_of_words,
    #                                                                                        unknown)
    # songs_library_of_emotion[song_index][20] = norm_disgust = normalize_emotion_value(disgust, total_number_of_words,
    #                                                                                   unknown)
    # songs_library_of_emotion[song_index][21] = norm_fear = normalize_emotion_value(fear, total_number_of_words, unknown)
    # songs_library_of_emotion[song_index][22] = norm_joy = normalize_emotion_value(joy, total_number_of_words, unknown)
    # songs_library_of_emotion[song_index][23] = norm_sadness = normalize_emotion_value(sadness, total_number_of_words,
    #                                                                                   unknown)
    # songs_library_of_emotion[song_index][24] = norm_surprise = normalize_emotion_value(surprise, total_number_of_words,
    #                                                                                    unknown)
    # songs_library_of_emotion[song_index][25] = norm_trust = normalize_emotion_value(trust, total_number_of_words,
    #                                                                                 unknown)
    songs_library_of_emotion[song_index][26] = norm_unknown = normalize_emotion_value(unknown, total_number_of_words,
                                                                                      unknown)
    songs_library_of_emotion[song_index][27] = norm_total_number_of_words = normalize_emotion_value(
        total_number_of_words, total_number_of_words, unknown)
    songs_library_of_emotion[song_index][28] = norm_emotional_impact_words = normalize_emotion_value(
        emotional_impact_words, total_number_of_words, unknown)
    songs_library_of_emotion[song_index][29] = norm_neutral_words = normalize_emotion_value(neutral_words,
                                                                                            total_number_of_words,
                                                                                            unknown)

    print("The song ", song_title, " (", artist, ", ", song_index, ") has: ")
    print("Valence: ", valence, "or normalized %", norm_valence)
    print("Arousal: ", arousal, "or normalized %", norm_arousal)
    print("Dominance: ", dominance, "or normalized %", norm_dominance)
    # print("Anticipation: ", anticipation, "or normalized %", norm_anticipation)
    # print("Disgust: ", disgust, "or normalized %", norm_disgust)
    # print("Fear: ", fear, "or normalized %", norm_fear)
    # print("Joy: ", joy, "or normalized %", norm_joy)
    # print("Sadness: ", sadness, "or normalized %", norm_sadness)
    # print("Surprise: ", surprise, "or normalized %", norm_surprise)
    # print("Trust: ", trust, "or normalized %", norm_trust)
    print("Unknown words: ", unknown, "or normalized %", norm_unknown)
    print("Total Number of Words: ", total_number_of_words, "or normalized %", norm_total_number_of_words)
    print("Words with emotional impact: ", emotional_impact_words, "or normalized %", norm_emotional_impact_words)
    print("Neutral words: ", neutral_words, "or normalized %", norm_neutral_words)

    #print(emotional_impact_words)

    return songs_library_of_emotion[song_index]


# test_song_id = 'ML1'
# test7 = lyrics_database.iloc['ML1'][3]


# results = (get_song_emotion(test_song_id, lyrics))




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


def tag_database(file_path):
    #lyrics_database = pd.read_csv(file_path)
    lyrics_database = pd.read_csv(file_path, encoding='latin-1')

    lyrics_database = lyrics_database.set_index("Index", drop=False)
    tagged_songs = {}

    for index, row in lyrics_database.iterrows():
        # print (index)
        # print (row)
        # print (row["Index"])
        song_index = row["Index"]
        song_title = row["Title"]
        song_artist = row["Artist"]
        song_lyrics = row["Lyrics"]
        songs_library_of_emotion[song_index] = [song_title, song_artist,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0]
        tagged_songs[song_index] = [song_title, song_artist,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0]

        # print(tagged_songs)
        get_song_emotion(song_index, song_lyrics)



        the_header = ["title", "artist","positive", "negative", "anger", "anticipation", "disgust", "fear",
                                      "joy", "sadness", "surprise", "trust", "unknown",
                                      "total_number_of_words", "emotional_impact_words", "neutral_words",
                                      "norm_positive", "norm_negative", "norm_anger", "norm_anticipation",
                                      "norm_disgust", "norm_fear", "norm_joy",
                                      "norm_sadness", "norm_surprise", "norm_trust", "norm_unknown",
                                      "norm_total_number_of_words", "norm_emotional_impact_words", "norm_neutral_words"]

        for i in range(2, 30):
            tagged_songs[song_index][i] = songs_library_of_emotion[song_index][i]

    (pd.DataFrame.from_dict(data=tagged_songs, orient='index')
     .to_csv('myresults_ev.csv', header=the_header))
    return tagged_songs

def tag_big_lyrics(file_path):
    #lyrics_database = pd.read_csv(file_path)
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
        songs_library_of_emotion[song_index] = [song_title, song_artist,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, song_genre, song_release_year]
        tagged_songs[song_index] = [song_title, song_artist,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, song_genre, song_release_year]

        # print(tagged_songs)
        get_song_emotion(song_index, song_lyrics)



        the_header = ["title", "artist","valence", "arousal", "dominance", "empty", "empty", "empty",
                                      "empty", "empty", "empty", "empty", "unknown",
                                      "total_number_of_words", "emotional_impact_words", "neutral_words",
                                      "norm_valence", "norm_arousal", "norm_dominance", "empty",
                                      "empty", "empty", "empty",
                                      "empty", "empty", "empty", "norm_unknown",
                                      "norm_total_number_of_words", "norm_emotional_impact_words", "norm_neutral_words", "genre", "year"]

        for i in range(2, 30):
            tagged_songs[song_index][i] = songs_library_of_emotion[song_index][i]

    (pd.DataFrame.from_dict(data=tagged_songs, orient='index')
     .to_csv('myresults_ev.csv', header=the_header))
    return tagged_songs

moby = "Who am I? Someone that's afraid to let go, uh You decide, if you're ever gonna let me know, yeah Suicide, if you ever try to let go, uh I'm sad and low, yeah, I'm sad I know, yeah Who am I? Someone that's afraid to let go, uh You decide, if you're ever gonna let me know, yeah Suicide, if you ever try to let go, uh I'm sad and low, yeah, I'm sad I know, yeah Who am I? Someone that's afraid to let go, uh You decide, if you're ever gonna let me know, yeah Suicide, if you ever try to let go, uh I'm sad and low, yeah, I'm sad I know, yeah Who am I? Someone that's afraid to let go, uh You decide, if you're ever gonna let me know, yeah Suicide, if you ever try to let go, uh I'm sad and low, yeah, I'm sad I know, yeah Who am I? Someone that's afraid to let go, uh You decide, if you're ever gonna let me know, yeah Suicide, if you ever try to let go, uh I'm sad and low, yeah, I'm sad I know, yeah Who am I? Someone that's afraid to let go, uh You decide, if you're ever gonna let me know, yeah Suicide, if you ever try to let go, uh I'm sad and low, yeah, I'm sad I know, yeah Who am I? Someone that's afraid to let go, uh You decide, if you're ever gonna let me know, yeah Suicide, if you ever try to let go, uh I'm sad and low, yeah, I'm sad I know, yeah Who am I? Someone that's afraid to let go, uh You decide, if you're ever gonna let me know, yeah Suicide, if you ever try to let go, uh I'm sad and low, yeah, I'm sad I know, yeah"
# moby_dick = "cheated my self"
print(get_song_emotion('ML1', moby))
#print(get_song_emotion('ML2', moby_dick))
#
# print(songs_library_of_emotion)
#
# song_lib = pd.DataFrame(songs_library_of_emotion)
# print(song_lib)
# print(song_lib.T)
path = '/Users/Vasilis/Desktop/Lennon/lyrics.csv'

# tag_big_lyrics(path)

# DISADVNATAGE: Negation dont count i am not happy happy happy

# melancholic dreams etc

