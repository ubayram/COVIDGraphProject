# Ulya Bayram
# ulyabayram@gmail.com
#

import subject_verb_object_extract_LB_0906_scispacy as sci # get Lamia's code
import spacy
import en_core_web_sm
import pandas as pd
import numpy as np
import nltk
#from codetiming import Timer

#t = Timer()

# use spacy small model
nlp = en_core_web_sm.load()

path_to_stopwords = 'nltk_stopwords.txt'
        
# Keep this block
with open(path_to_stopwords, 'r') as file:
    stopwords = file.readlines()
for word in stopwords:
    nlp.vocab[word].is_stop = True

def cleanText(curr_text):

    curr_text = curr_text.lower()
    return curr_text.split(' . ')

def collectSaveEdges(df, pd_eval, row_indices, date_flag):

    for i_row in row_indices:
        list_sentences = cleanText(df.full_text[i_row])

        if date_flag:
            curr_date = df.date[i_row]
        else:
            curr_date = df.year[i_row]

        curr_tuples = []
        #t.start()
        for curr_sent in list_sentences:
            if len(curr_sent.split(' ')) > 3: # make sure a sentence has at least 4 words - to eliminate noise
                #print('\n'+curr_sent)
                curr_tuples += sci.extract_link(nlp(curr_sent))

        #t.stop()
        pd_eval = pd_eval.append(pd.DataFrame({'filename' : df.fullname[i_row], 'list_of_edges': [curr_tuples], 'timestamp': curr_date}))
        print('Processing row ' + str(i_row))

    return pd_eval

if __name__ == '__main__':

    # read the corpus
    all_data = pd.read_csv('full_cord19_texts.csv')

    print('Processing before 2020, first half')
    # list of rows where years are before 2020
    row_indices_pre = all_data[all_data['year'] < 2020].index.tolist()
    half_index = int(len(row_indices_pre)/2)
    row_indices_pre = row_indices_pre[:half_index]
    empty_list = []
    pd_eval = pd.DataFrame(data={'filename' : empty_list, 'list_of_edges': empty_list, 'timestamp': empty_list})
    pd_eval = collectSaveEdges(all_data, pd_eval, row_indices_pre, False)
    pd_eval.to_csv('list_of_edges_pre2020_first_half.csv')
    del pd_eval

    '''
    new_df = pd.read_csv('list_of_edges_pre2020.csv', converters={'list_of_edges': eval})
    print(len(new_df.list_of_edges[0]))
    print(new_df.list_of_edges[0])
    print(new_df.list_of_edges[0][0])
    '''