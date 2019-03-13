# A class to create a custom corpus.

import xlrd
import os

# Make three new directories for the corpus: main folder, positive, negative
corpusdir = '/Users/Vasilis/Desktop/Lennon/lyrics_custom_corpus'
corpusdir_pos = corpusdir + '/pos'
corpusdir_neg = corpusdir + '/neg'


def replace_with_underscores(cell):
    return cell.value.replace(" ", "_")


if not os.path.isdir(corpusdir):
    os.mkdir(corpusdir)
    os.mkdir(corpusdir_pos)
    os.mkdir(corpusdir_neg)

# Provide the paths to the neg and pos Excel files.
path_neg = "/Users/Vasilis/Desktop/Lennon/test7neg.xls"
path_pos = "/Users/Vasilis/Desktop/Lennon/test7pos.xls"

# Read negative and positive lyrics from Excel and write them to .txt files

neg = xlrd.open_workbook(path_neg)
pos = xlrd.open_workbook(path_pos)
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
