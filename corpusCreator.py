import xlrd
import os

# Make new dir for the corpus.
corpusdir = '/Users/Vasilis/Desktop/Lennon/lyrics_custom_corpus'
corpusdir_pos = corpusdir + '/pos'
corpusdir_neg = corpusdir + '/neg'


def replace_with_underscores(cell):
    return cell.value.replace(" ", "_")


if not os.path.isdir(corpusdir):
    os.mkdir(corpusdir)
    os.mkdir(corpusdir_pos)
    os.mkdir(corpusdir_neg)

neg = xlrd.open_workbook("/Users/Vasilis/Desktop/Lennon/test7neg.xls")
pos = xlrd.open_workbook("/Users/Vasilis/Desktop/Lennon/test7pos.xls")
sh_neg = neg.sheet_by_index(0)
sh_pos = pos.sheet_by_index(0)

for row in sh_neg.get_rows():
    index = replace_with_underscores(row[0])
    title = replace_with_underscores(row[2])
    lyrics = row[4].value

    filename = os.path.join(corpusdir_neg, index + "-" + title + ".txt")
    with open(filename, 'w') as f:
        f.write(lyrics)

for row in sh_pos.get_rows():
    index = replace_with_underscores(row[0])
    title = replace_with_underscores(row[2])
    lyrics = row[4].value

    filename = os.path.join(corpusdir_pos, index + "-" + title + ".txt")
    with open(filename, 'w') as f:
        f.write(lyrics)

# solve bug


# import os
# from nltk.corpus.reader.plaintext import PlaintextCorpusReader
#
# corpusdir = '/Users/vasilis/Desktop/Lennon/lyrics_custom_corpus'  # Directory of corpus.
#
# newcorpus = PlaintextCorpusReader(corpusdir, '.*')
#
# print(newcorpus)
