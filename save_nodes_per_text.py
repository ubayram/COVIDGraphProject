# Ulya Bayram
# ulyabayram@gmail.com
#

import subject_verb_object_extract_LB_2406_scispacy as sci # get Lamia's code
import help_NER_UB as help_ner
import spacy
import en_core_web_sm
import pandas as pd
import numpy as np
import nltk
#from codetiming import Timer
import time

#t = Timer()

# use spacy small model
nlp = en_core_web_sm.load()

stopwords = help_ner.getStopwords()

for word in stopwords:
    nlp.vocab[word].is_stop = True

def cleanText(curr_text):

    curr_text = curr_text.lower()
    return curr_text.split(' . ')

def collectSaveEdges(df, pd_eval, row_indices, date_flag):

    for i_row in row_indices:
        curr_text = df.full_text[i_row]
        if not help_ner.isEnglish(curr_text): # skip non-English data rows
            continue
        list_sentences = cleanText(curr_text)
        del curr_text

        if date_flag:
            curr_date = df.date[i_row]
        else:
            curr_date = df.year[i_row]

        curr_tuples = []
        #t.start()
        t0 = time.time()
        for curr_sent in list_sentences:
            if len(curr_sent.split(' ')) > 4: # make sure a sentence has at least 4 words - to eliminate noise
                #print('\n'+curr_sent)
                
                links = sci.extract_link(nlp(curr_sent))
                #if len(links): # Ulya: if links returns an empty list, curr_tuples won't be affected. I'm removing this if for speed
                # print(links)
                curr_tuples += links

        #t.stop()
        t1 = time.time()
        print(t1-t0)
                
        pd_eval = pd_eval.append(pd.DataFrame({'filename' : df.fullname[i_row], 'list_of_edges': [curr_tuples], 'timestamp': curr_date}))
        print('Processing row ' + str(i_row))

    return pd_eval

if __name__ == '__main__':

    # read the corpus
    all_data = pd.read_csv('/Users/mac/Documents/GitHub/COVIDGraphProject/data/full_cord19_texts.csv')

    print('Processing before 2020, first half')
    # list of rows where years are before 2020
    row_indices_pre = all_data[all_data['year'] < 2020].index.tolist()
    half_index = int(len(row_indices_pre)/2)
    row_indices_pre = row_indices_pre[:half_index]
    empty_list = []
    pd_eval = pd.DataFrame(data={'filename' : empty_list, 'list_of_edges': empty_list, 'timestamp': empty_list})
    pd_eval = collectSaveEdges(all_data, pd_eval, row_indices_pre, False)
    pd_eval.to_csv('list_of_edges_pre2020_first_half_new.csv')
    del pd_eval

    '''
    new_df = pd.read_csv('list_of_edges_pre2020.csv', converters={'list_of_edges': eval})
    print(len(new_df.list_of_edges[0]))
    print(new_df.list_of_edges[0])
    print(new_df.list_of_edges[0][0])
    '''