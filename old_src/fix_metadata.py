# Ulya Bayram
# ulyabayram@gmail.com
# NLP processes happens here
import json
import glob
import nltk
import pandas as pd
import math

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
        return text_.replace(',', ' ')

def fixAbstract(text_):
    text_ = text_.lower()
    table = str.maketrans('', '', '!"#$&()*+,/:;<=>?@[\]^_`{|}~') # keep % sign, delete commas, keep . and ?
    text_ = text_.translate(table)
    text_ = text_.replace('–', ' ')
    text_ = text_.replace('-', ' ')
    text_ = text_.replace('  ', ' ')
    text_ = text_.replace('   ', ' ')

    if  isinstance(text_.split(' ')[-1], int): # delete the page number if exists
        del text_.split(' ')[-1]

        text_ = ' '.join(text_)
    return text_

def abstractCheckWriteData(file_to_write, meta_data, obj, index_, abstract_list, new_timestamp, abstract_check):
    # abstract section exists in the jason and abstract isn't empty or a single number
    if 'abstract' in obj.keys() and len(obj['abstract']) > 0:
        processed_abstract = obj['abstract'][0]['text']

        if not isNaN(abstract_list[index_]):
            meta_abstract = abstract_list[index_]
        else:
            meta_abstract = ''

        if len(processed_abstract.split(' ')) > len(meta_abstract.split(' ')): # if abstract is fine in json, read and write it
            orig_abstract = processed_abstract
            processed_abstract = processed_abstract.replace('\n', ' ')
            processed_abstract = fixAbstract(processed_abstract)

            if orig_abstract not in abstract_check:
                file_to_write.write(curr_sha + ',' + fixData(meta_data.source_x[index_]) + ',' + processed_abstract + ',' + new_timestamp + '\n')
                abstract_check.append(orig_abstract)

        else: # abstract in metadata
            if not isNaN(abstract_list[index_]):
                # something wrong with abstract in json - use the abstract in the meta_data
                if abstract_list[index_] not in abstract_check:
                    file_to_write.write(curr_sha + ',' + fixData(meta_data.source_x[index_]) + ',' + fixAbstract(abstract_list[index_]) + ',' + new_timestamp + '\n')
                    abstract_check.append(abstract_list[index_])

    else: # json doesn't have abstract, no other option, go with the abstract in the metadata
        if not isNaN(abstract_list[index_]):
            if abstract_list[index_] not in abstract_check:
                file_to_write.write(curr_sha + ',' + fixData(meta_data.source_x[index_]) + ',' + fixAbstract(abstract_list[index_]) + ',' + new_timestamp + '\n')
                abstract_check.append(abstract_list[index_])

    return abstract_check

def fixTimestamp(curr_sha, curr_time, list_fixable_shas, data_timestamps):

    if curr_sha in list_fixable_shas:
        i_ = list_fixable_shas.index(curr_sha)
        new_time = data_timestamps.date_API[i_]
        if isNaN(new_time):
            return 'skip'
        else:
            return new_time.replace('/', '-')
    else:
        return curr_time
#########################################################################
# ------------------------- main part here -----------------------------

# the folders to read the data from
sub_folders = ['biorxiv_medrxiv/pdf_json/', 'comm_use_subset/pdf_json/',
             'custom_license/pdf_json/', 'noncomm_use_subset/pdf_json/',
             'comm_use_subset/pmc_json/', 'custom_license/pmc_json/', 'noncomm_use_subset/pmc_json/']

# create the empty csv file
file_to_write = open('processed_dataset.csv', 'w')
file_to_write.write('sha,source_x,abstract,publish_time\n')

#file_for_api = open('papers_to_scrape_by_API_2.csv', 'w')
#file_for_api.write('sha,source_x,title,doi,publish_time,url\n')

# read and fill the missing timestamps
data_timestamps = pd.read_csv('papers_to_scrape_for_timestamps_with_dates.csv')
fixable_shas = list(data_timestamps.sha)

# read the metadata file
meta_data = pd.read_csv('metadata.csv')
sha_dict = fixShaList(list(meta_data.sha))
abstract_list = list(meta_data.abstract)
title_list = list(meta_data.title)

abstract_check = []

# loop over the folders
for i_folder in sub_folders:

    # get all the filenames in the current directory
    set_of_filenames = glob.glob(i_folder + '*.json')

    # read the json files
    for i_json in set_of_filenames:  
        fo_text = open(i_json, 'r').read()
        obj = json.loads(fo_text)

        # get the sha only
        curr_sha = getShaOnly(i_json)
        if curr_sha in sha_dict.keys():
            index_ = sha_dict[curr_sha]
            curr_time = fixData(meta_data.publish_time[index_])
            new_timestamp = fixTimestamp(curr_sha, curr_time, fixable_shas, data_timestamps)
            if 'skip' not in new_timestamp:
                abstract_check = abstractCheckWriteData(file_to_write, meta_data, obj, index_, abstract_list, new_timestamp, abstract_check)
        else: # current sha doesn't exist in the meta_data file
            # get the title of the current paper
            title_ = obj['metadata']['title']
            # match the title with those in the meta_data
            if title_ in title_list:
                # find the index
                index_ = title_list.index(title_)
                curr_time = fixData(meta_data.publish_time[index_])
                new_timestamp = fixTimestamp(curr_sha, curr_time, fixable_shas, data_timestamps)
                if 'skip' not in new_timestamp:
                    abstract_check = abstractCheckWriteData(file_to_write, meta_data, obj, index_, abstract_list, new_timestamp, abstract_check)

            #else: # title doesn't match either or doesn't exist
            #    file_for_api.write(curr_sha + ',' + fixData(meta_data.source_x[index_]) + ',' + fixData(meta_data.title[index_]) + ',' + fixData(meta_data.doi[index_]) + ',' +\
            #        fixData(meta_data.publish_time[index_]) + ',' + fixData(meta_data.url[index_]) + '\n')

file_to_write.close()