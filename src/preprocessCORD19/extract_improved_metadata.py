# Author: Ulya Bayram
# Contact: ulyabayram@gmail.com
# This code fills the empty cells in the original metadata file using the original texts
# Applying an extensive set of operations - as metadata has plenty of empty cells
# Returns a new metadata dataframe and another dataframe that is the subset of the new metadata with bad timestamps
import glob
import pandas as pd
import json
import numpy as np
from fuzzywuzzy import process, fuzz

def collectOnlyFilenames(files_list):

    list_filenames = []
    for curr_file in files_list:
        if not isNaN(curr_file):
            tmp1 = curr_file.split('/')[-1]
            tmp2 = tmp1.split('.')[:-1]
            tmp2 = '.'.join(tmp2)

            list_filenames.append(tmp2)
        else:
            list_filenames.append(curr_file)

    return list_filenames

def stripSingleFilename(file_):

    tmp1 = file_.split('/')[-1]
    tmp2 = tmp1.split('.')[:-1]
    tmp2 = '.'.join(tmp2)

    return tmp2

def getUniqueFilename(curr_file):
    tmp1 = curr_file.split('/')[-2:]
    return '_'.join(tmp1)

def isNaN(num):
    return num != num

def getAuthors(obj):
    num_authors = len(obj['metadata']['authors'])
    auth_list = []
    for i_auth in range(num_authors):
        curr_dict = obj['metadata']['authors'][i_auth]
        auth_list.append( curr_dict['last'] + ', ' + curr_dict['first'] )
    
    return '; '.join(auth_list)

def getFullText(obj):
    num_texts = len(obj['body_text'])
    curr_body = []
    for i_text in range(num_texts):
        curr_text = obj['body_text'][i_text]['text']
        curr_body.append(curr_text)

    return ' '.join(curr_body)

def getPaperInfo(obj):
    title_ = obj['metadata']['title']

    if 'abstract' in obj.keys() and len(obj['abstract']) > 0:
        abstract_ = obj['abstract'][0]['text']
    else:
        abstract_ = ''

    return title_, abstract_, getAuthors(obj)

def writeText2File(full_text, filename, savedir):
    fo = open(savedir + 'processed_texts/' + filename + '.txt', 'w')
    fo.write(full_text)
    fo.close()

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

def fromIndicesReturnBest(row_indices, time_list, doi_list):

    if len(row_indices) > 1:
        timestamps = []
        dois = []

        for i_row in row_indices:
            timestamps.append(time_list[i_row])
            dois.append(doi_list[i_row])

        # return the index for the one with largest timestamp data
        candidate_time = max(enumerate(timestamps), key=lambda x: len(x[1]))[1]
        candidate_doi = max(enumerate(dois), key=lambda x: len(x[1]))[1]

        return candidate_time, candidate_doi
    else:
        i_row = row_indices[0]
        #print(i_row)
        return time_list[i_row], doi_list[i_row] # TypeError: list indices must be integers or slices, not numpy.float64

def fromAuthorIndicesReturnBest(row_indices, time_list, doi_list, curr_title, curr_abstract, meta_data):

    if len(row_indices) > 1:
        timestamps = []
        dois = []

        if not isNaN(curr_title) and len(curr_title)>3:
            short_title_list = list(meta_data.title[row_indices]) # list(title_list[row_indices])
            highest_match = process.extractOne(curr_title, short_title_list, scorer=fuzz.token_set_ratio)

            if highest_match[1] > 99:
                i_row = row_indices[short_title_list.index(highest_match[0])]
                return time_list[i_row], doi_list[i_row]

        if len(curr_abstract) > 3:
            short_abstract_list = list(meta_data.abstract[row_indices]) # list(abstract_list[row_indices])
            highest_match = process.extractOne(curr_abstract, short_abstract_list, scorer=fuzz.token_set_ratio)

            if highest_match[1] > 99:
                i_row = row_indices[short_abstract_list.index(highest_match[0])]
                return time_list[i_row], doi_list[i_row]
        
        for i_row in row_indices:
            timestamps.append(time_list[i_row])
            dois.append(doi_list[i_row])

        #print('The worst way of doing this')
        # return the index for the one with largest timestamp data
        candidate_time = max(enumerate(timestamps), key=lambda x: len(x[1]))[1]
        candidate_doi = max(enumerate(dois), key=lambda x: len(x[1]))[1]

        return candidate_time, candidate_doi
    else: # if the author list matches only once, then assume it's the correct paper's metadata
        i_row = row_indices[0]
        return time_list[i_row], doi_list[i_row]

def getMetadata(meta_data, sha_dict, curr_title, curr_abstract, curr_authors, curr_sha):
    #abstract_list = list(meta_data.abstract)
    #title_list = list(meta_data.title)
    time_list = list(meta_data.publish_time)
    doi_list = list(meta_data.doi)
    pdf_json_list = pd.DataFrame({'col': collectOnlyFilenames(list(meta_data.pdf_json_files))})
    pmc_json_list = pd.DataFrame({'col': collectOnlyFilenames(list(meta_data.pmc_json_files))})
    #author_list = list(meta_data.authors)

    # order of matching the read files to the information in metadata.csv file
    #1) see if the sha matches - if yes, collect the information and return it, else:
    if curr_sha in sha_dict.keys():
        i_row = sha_dict[curr_sha]
        curr_timestamp = time_list[i_row]
        curr_doi = doi_list[i_row]

        return curr_timestamp, curr_doi
    #2) see if the directory included name matches - if yes collect the information and return it, else:
    elif curr_sha in list(pdf_json_list.col) or curr_sha in list(pmc_json_list.col):
        if curr_sha in list(pdf_json_list.col): #Index_label = df[df['Updated Price']>1000].index.tolist()
            row_indices1 = pdf_json_list[pdf_json_list['col'] == curr_sha].index.tolist()   #np.where(pdf_json_list == curr_sha)[0] # get rid of numpy here - it has problems
        else:
            row_indices1 = []
        if curr_sha in list(pmc_json_list.col):
            row_indices2 = pmc_json_list[pmc_json_list['col'] == curr_sha].index.tolist() #np.where(pmc_json_list == curr_sha)[0]
        else:
            row_indices2 = []
        row_indices = np.concatenate((row_indices1, row_indices2), axis=0)
        row_indices = row_indices.astype('int32')
        curr_timestamp, curr_doi = fromIndicesReturnBest(row_indices, time_list, doi_list)

        return curr_timestamp, curr_doi
    #3) see if the title matches - if yes, collect the information and return it, else:
    elif len(curr_title) > 0 and curr_title in list(meta_data.title):
        row_indices = meta_data[meta_data['title'] == curr_title].index.tolist() # np.where(title_list == curr_title)[0] # there could be more than one
        curr_timestamp, curr_doi = fromIndicesReturnBest(row_indices, time_list, doi_list)

        return curr_timestamp, curr_doi
    #4) see if the abstract matches - if yes, collect the info and return, else:
    elif len(curr_abstract) > 3 and curr_abstract in list(meta_data.abstract):
        row_indices = meta_data[meta_data['abstract'] == curr_abstract].index.tolist() #np.where(np.array(abstract_list) == curr_abstract)[0] # there could be more than one
        curr_timestamp, curr_doi = fromIndicesReturnBest(row_indices, time_list, doi_list)

        return curr_timestamp, curr_doi
    #5) see if the author list matches - among the returned papers find current title contains the given title or the abstract
    elif curr_authors in list(meta_data.authors): # this is the solution for 70,000 remaining unmatched metadata - sensitive work required
        row_indices = meta_data[meta_data['authors'] == curr_authors].index.tolist() # np.where(author_list == curr_authors)[0] # there could be more than one
        curr_timestamp, curr_doi = fromAuthorIndicesReturnBest(row_indices, time_list, doi_list, curr_title, curr_abstract, meta_data)

        return curr_timestamp, curr_doi
    else:
        # No other choice but to let it go
        return '', ''

def collectBadTimestamps(metadata):
    list_shas = []
    list_foldernames = []
    list_titles = []
    list_timestamps = []
    list_dois = []
    for i_row in range(len(metadata.sha)):

        if len(metadata['publish_time'][i_row]) < 5:
            list_shas.append(metadata['sha'][i_row])
            list_titles.append(metadata['title'][i_row])
            list_foldernames.append(metadata['fullnames'][i_row])
            list_timestamps.append(metadata['publish_time'][i_row])
            list_dois.append(metadata['doi'][i_row])

    df = pd.DataFrame({'sha': list_shas, 'fullnames' : list_foldernames, 'title': list_titles, 'publish_time': list_timestamps, 'doi':list_dois})

    return df

def extractNewImprovedMetadata(datadir, savedir, metadatadir):
    """ Reads the sha list from the original metadata file which is closest thing to associate metadata rows to json files
        Then reads the authors list, title, abstract and the filenames from the json files
        Tries to match those in the metadata rows - which is the only source of information for timestamps and doi's
        Fills the empty cells in the original metadata file by the matching operations (including fuzzy text matching - json titles sometimes have additional noise)
        Returns a brand new metadata file that is improved and filled as much as possible
        Also returns a list of bad timestamps for some can be recovered from doi's - if they have non-empty doi's in the original metadata
        Contact me (Ulya Bayram) if there are problems.
    """
    # read the metadata file
    meta_data = pd.read_csv(metadatadir + 'metadata.csv', dtype=str)
    sha_dict = fixShaList(list(meta_data.sha))

    list_filenames = glob.glob(datadir + '*.json')

    new_list_filenames = []
    new_list_titles = []
    new_list_abstracts = []
    new_list_authors = []
    new_list_timestamps = []
    new_list_dois = []
    list_full_names = []
    #print(len(list_filenames))
    i_count = 0
    for curr_file in list_filenames:
        fo_text = open(curr_file, 'r').read()
        obj = json.loads(fo_text)

        curr_sha = stripSingleFilename(curr_file)
        filename = getUniqueFilename(curr_file)
        full_text = getFullText(obj)
        writeText2File(full_text, filename, savedir)
        list_full_names.append(filename)
        curr_title, curr_abstract, curr_authors = getPaperInfo(obj)

        new_list_filenames.append(curr_sha)
        new_list_titles.append(curr_title)
        new_list_abstracts.append(curr_abstract)
        new_list_authors.append(curr_authors)

        curr_timestamp, curr_doi = getMetadata(meta_data, sha_dict, curr_title, curr_abstract, curr_authors, curr_sha)
        new_list_timestamps.append(curr_timestamp)
        new_list_dois.append(curr_doi)

        #print(i_count)
        i_count += 1

    # write the new metadata.csv file using pandas dataframe
    # ('sha,title,abstract,authors,publish_time,doi\n')
    new_metadata_df = pd.DataFrame({'sha': new_list_filenames, 'fullnames' : list_full_names, 'title': new_list_titles, 'abstract':new_list_abstracts, 'authors':new_list_authors,\
        'publish_time': new_list_timestamps, 'doi':new_list_dois})

    #df.to_csv('new_metadata.csv', index=False)

    bad_times_df = collectBadTimestamps(new_metadata_df)

    return new_metadata_df, bad_times_df