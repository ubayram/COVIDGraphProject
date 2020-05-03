# Ulya Bayram
# ulyabayram@gmail.com
# NLP processes happens here
import json
import glob
import nltk
import pandas as pd
import math

# lowercase conversion
# tokenize by sentences - each sentence to be separated by comma
# remove punctuations here except for '
# join the sentences by , for the csv file
def preprocessText(text_):

    text_ = text_.lower()

    list_sentences = nltk.sent_tokenize(text_)

    table = str.maketrans('', '', '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
    stripped_sentences = [w.translate(table) for w in list_sentences]

    return ','.join(stripped_sentences)

def getShaOnly(filename):
    tmp = filename.split('/')[-1]

    return tmp.replace('.json', '')

# handle having multiple sha's in a single line
def fixShaList(sha_list):
    sha_dict = {}
    i_row = 0
    for i_sha in sha_list:
        if not isNaN(i_sha):
            if ';' in i_sha:
                curr_list = i_sha.split(';')
                for curr_item in curr_list:
                    sha_dict[curr_item.replace(' ', '')] = i_row
            else:
                sha_dict[i_sha] = i_row
        
        i_row += 1
    return sha_dict

def isNaN(num):
    return num != num

def fixData(text_):
    if isNaN(text_):
        return str(text_)
    else:
        return str(text_)

#########################################################################
# ------------------------- main part here -----------------------------

# create the empty csv file
meta_data = pd.read_csv('processed_metadata_2.csv')

file_for_api = open('papers_to_scrape_for_timestamps.csv', 'w')
file_for_api.write('sha,source_x,title,doi,publish_time,url\n')

# read the metadata file
sha_list = list(meta_data.sha)
time_stamps = list(meta_data.publish_time)
print(len(sha_list))
print(len(time_stamps))

# loop over the folders
for i in range(len(time_stamps)):
    curr_time = time_stamps[i]

    if len(curr_time.split('-')) < 3:
        print(fixData(meta_data.url[i]))
        file_for_api.write(sha_list[i] + ',' + fixData(meta_data.source_x[i]) + ',' + fixData(meta_data.title[i]) + ',' +\
            fixData(meta_data.doi[i]) +',' + curr_time +',' + fixData(meta_data.url[i]) + '\n')

file_for_api.close()