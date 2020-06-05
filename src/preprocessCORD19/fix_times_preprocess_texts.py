# Author: Ulya Bayram
# Contact: ulyabayram@gmail.com
# This code applies the final operations on the data collection
# Includes preprocessing the texts
# Removing repeated papers from the collection by using text comparison operations
# Also filling the retrieved new timestamps and eliminating those that don't qualify
# Final output is a file containing the full text bodies and proper timestamps that allows us to conduct an evolutionary analysis
import pandas as pd
import numpy as np
import regex as re

def isNaN(num):
    return num != num

def fixDates(new_time):
    if not isNaN(new_time):
        return new_time.replace('/', '-')
    else:
        return ''

def getAllFullTexts(metadata):

    list_filenames = []
    list_full_texts = []
    for filename in list(metadata['fullnames']):
        fo = open('processed_texts/' + filename + '.txt', 'r')
        full_text = fo.read()
        fo.close()

        # make sure it's a valid, normally sized text body
        num_sentences = len(full_text.split('.'))

        if num_sentences > 5: # a text should have at least 5 sentences
            list_filenames.append(filename)
            list_full_texts.append(full_text)
    
    return list_filenames, list_full_texts

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

def collectProperTimestamp(current_timestamp, filename, api_data):

    if len(current_timestamp) > 4: # ideal timestamp data
        year_ = current_timestamp.split('-')[0]
        return True, year_, current_timestamp
    else: # timestamp is short, so it should ideally be present in the api list
        #print(filename)
        #print(current_timestamp)
        api_row = api_data[api_data['fullnames'] == filename].index.tolist()[0] # there should be only one match
        api_time = fixDates(api_data.date_API[api_row])

        if len(api_time) > 0:
            year_ = api_time.split('-')[0]
            return True, year_, api_time
        else: # this means that current timestamp is only a year, and API couldn't find anything better for it
            # check if the year is pre-2020 - round timestamp to year-01-01 and is qualified to go
            if int(current_timestamp) < 2020 and int(current_timestamp) > 1900: # make sure it's not empty or NaN
                return True, current_timestamp, str(current_timestamp + '-01-01')
            elif int(current_timestamp) >= 2020: # we needed the exact date for these papers, but API returned nothing
                return False, '', '' # This paper is unqualified due to lack of a proper timestamp
            else:
                raise ValueError('Invalid timestamp encountered.')

def preprocessText(text_):
    #print(text_)
    #text_ = text_.lower()
    # remove things within () or [], add a space instead
    text_ = re.sub(r'\([^)]*\)', ' ', text_)
    text_ = re.sub(r'\[.*?\]', ' ', text_)
    text_ = re.sub(r'[^a-zA-Z0-9\.\-\,\?\!\;\:\% ]+', ' ', text_)
    text_ = re.sub('\.+', ' . ', text_) # not likely, but if there are "..." handle that
    text_ = text_.replace('.', ' . ')
    text_ = text_.replace('?', ' ? ')
    text_ = text_.replace('!', ' ! ')
    text_ = text_.replace(';', ' ; ')
    text_ = text_.replace(':', ' : ')
    text_ = re.sub(' +', ' ', text_) # single space between each token including some punctuations
    
    # fix those numbers separated by space, such as 8 .2 merged into 8.2
    text_ = re.sub(r'(\d)\s+\.(\d)', r'\1.\2', text_) # this and the one below are redundant - unnecessary since none will match but won't hurt to keep
    text_ = re.sub(r'(\d)\.\s+(\d)', r'\1.\2', text_)
    text_ = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text_)
    text_ = re.sub(r'(\d)\,\s+(\d)', r'\1\2', text_)
    text_ = re.sub(r'(\d)\s+\,(\d)', r'\1\2', text_)
    text_ = re.sub(r'(\d)\s+\,\s+(\d)', r'\1\2', text_)

    text_ = text_.replace(',', ' ')
    text_ = re.sub(' +', ' ', text_) # single space between each token including some punctuations

    return text_

def updateEvalKeeper(list_eval_texts, pd_eval, curr_text):

    if len(list_eval_texts) < 75:
        list_eval_texts.append(curr_text)
    else:
        pd_eval = pd_eval.append(pd.DataFrame({'A':list_eval_texts}), ignore_index=True)
        list_eval_texts = []

    return list_eval_texts, pd_eval

# -----------------------------------------------------------------
def fixCollection(metadata, api_data, savedir):
    """ Reads the complete set of text bodies from the folder that was previously created and filled - saves them in a temp dataframe
        Eliminates short text files from the dataset
        Then, for each text body surviving the elimination, all repetitions and their proper timestamps are found - best (full) timestamp is returned
        Finally, the unique text bodies are preprocessed, and written into a csv file together with their timestamps
        Contact me (Ulya Bayram) if there are problems.
    """

    #metadata = pd.read_csv('new_metadata.csv')
    list_filenames, list_full_texts = getAllFullTexts(metadata)
    texts_df = pd.DataFrame({'filename': list_filenames, 'text': list_full_texts})
    del list_filenames, list_full_texts

    #api_data = pd.read_csv('data_with_timestamps.csv')

    # Lazy way of handling pandas related RAM problems
    # so, the best option is to write line by line. Since it's csv, it can be read by pandas - so problem solved
    fw = open(savedir + 'full_cord19_texts.csv', 'w')
    fw.write('sha,fullname,full_text,year,date')

    # lists to store already evaluated texts
    list_eval_texts = []
    pd_eval = pd.DataFrame(data={'A' : list_eval_texts})
    # lists to be written in the final file
    #list_texts_2_write = []
    #list_full_names = []
    #list_sha = []
    #list_timestamps = []
    #list_years = []

    i_count = 0
    for filename in list(metadata['fullnames']):

        print(i_count)
        i_count += 1
        # read the text body
        fo = open(savedir + 'processed_texts/' + filename + '.txt', 'r')
        full_text = fo.read()
        fo.close()

        if full_text in list(pd_eval.A) or full_text in list_eval_texts or len(full_text.split('.')) < 5:
            continue

        # find its occurrences in the full list of texts (DF)
        row_indices_text_list = texts_df[texts_df['text'] == full_text].index.tolist()
        matching_files = list(texts_df.filename[row_indices_text_list])

        if len(matching_files) == 1:
            row_indices_metadata = metadata[metadata['fullnames'] == matching_files[0]].index.tolist()
            i_row = row_indices_metadata[0]
            # check the timestamp data, whether it qualifies to be included in the dataset
            current_timestamp = metadata.publish_time[i_row]

            is_qualified, year_, timestamp_ = collectProperTimestamp(current_timestamp, filename, api_data)

            if is_qualified:
                # add to evaluated texts list
                list_eval_texts, pd_eval = updateEvalKeeper(list_eval_texts, pd_eval, full_text)

                # pre-process the text data
                processed_text = preprocessText(full_text)
                #print(processed_text)
                fw.write('\n' + metadata.sha[i_row] + ',' + filename + ',' + processed_text + ',' + str(year_) + ',' + timestamp_)

        elif len(matching_files) > 1: # find timestamps of this text, select the best
            # add to evaluated texts list
            list_eval_texts, pd_eval = updateEvalKeeper(list_eval_texts, pd_eval, full_text)

            row_indices_metadata = metadata[metadata['fullnames'] == matching_files[0]].index.tolist()
            tmp_list_timestamps = []
            tmp_list_years = []
            for i_row in row_indices_metadata:
                # check the timestamp data, whether it qualifies to be included in the dataset
                current_timestamp = metadata.publish_time[i_row]

                is_qualified, year_, timestamp_ = collectProperTimestamp(current_timestamp, filename, api_data)

                if is_qualified:
                    tmp_list_timestamps.append(timestamp_)
                    tmp_list_years.append(timestamp_)
            
            if len(list(set(tmp_list_timestamps))) > 1:
                raise ValueError('Multiple different timestamps for the same paper.')
            else:
                # pre-process the text data
                processed_text = preprocessText(full_text)
                #print(processed_text)
                fw.write('\n' + metadata.sha[row_indices_metadata[0]] + ',' + filename + ',' + processed_text + ',' + str(tmp_list_years[0]) + ',' + tmp_list_timestamps[0])

        #else: # skip empty texts
        #    print('Skip empty text')
        #if i_count == 1000:
        #    break
    # save the dataframe
    #print('Saving the DF')
    #df2 = pd.DataFrame({'sha': list_sha, 'fullname' : list_full_names, 'full_text': list_texts_2_write, 'year': list_years, 'date': list_timestamps})
    #df2.to_csv('full_cord19_texts.csv', index=False, mode='a')
    #print('Saved the DF')
    fw.close()
    # The End of Preprocessing :)