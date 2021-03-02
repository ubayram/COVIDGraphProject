#Author: Ulya Bayram
#email : ulya.bayram@comu.edu.tr
#
#------------------------------------------------------------------------------------------------------
#
#The content of this project is licensed under the MIT license. 2021 All rights reserved.
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
#and associated documentation files (the "Software"), to deal with the Software without restriction, 
#including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
#and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
#subject to the following conditions:
#
#Redistributions of source code must retain the above License notice, this list of conditions and 
#the following disclaimers.
#
#Redistributions in binary form must reproduce the above License notice, this list of conditions and 
#the following disclaimers in the documentation and/or other materials provided with the distribution. 
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
#LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
#IN NO EVENT SHALL THE CONTRIBUTORS OR LICENSE HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
#WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
#OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
#
#------------------------------------------------------------------------------------------------------
#
#These code are writen for a research project, published in OIR. If you use any of them, please cite:

#Ulya Bayram, Runia Roy, Aqil Assalil, Lamia Ben Hiba, 
#"The Unknown Knowns: A Graph-Based Approach for Temporal COVID-19 Literature Mining", 
#Online Information Review (OIR), COVID-19 Special Issue, 2021.
#
#------------------------------------------------------------------------------------------------------
# This is where the node and edge extraction as well as the edge weight computation happens

import subject_verb_object_extraction as sci # get Lamia's code
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

def eliminateImproperDatesPost2020(row_indices_pre, df):

    row_indices_pre_clean = []
    filenames_ = [] # store them too for just in case row orders gets messed up
    for i_row in row_indices_pre: # clean up the indices - some dates are messed up
        curr_date = df.date[i_row]
        year_ = int(curr_date.split('-')[0])

        if year_ >= 2020:
            row_indices_pre_clean.append(i_row)
            filenames_.append(df.fullname[i_row])

    return row_indices_pre_clean, filenames_ # after done, write these indices, will be necessary to Aqil.

def extract_nodes_edges(input_dir):

    # read the corpus
    all_data = pd.read_csv(input_dir)

    print('Processing after 2020, so COVID-19 specific papers only')
    # list of rows where years are after 2020
    row_indices_pre_ = all_data[all_data['year'] >= 2020].index.tolist()
    row_indices_pre, filenames_ = eliminateImproperDatesPost2020(row_indices_pre_, all_data)

    #new_df = pd.DataFrame({'filename': filenames_, 'i_rows': row_indices_pre})
    #new_df.to_csv('post2020_qualifying_filenames_row_indices.csv')

    #half_index = int(len(row_indices_pre)/2)
    #row_indices_pre = row_indices_pre[half_index:]
    empty_list = []
    pd_eval = pd.DataFrame(data={'filename' : empty_list, 'list_of_edges': empty_list, 'timestamp': empty_list})
    pd_eval = collectSaveEdges(all_data, pd_eval, row_indices_pre, True)
    pd_eval.to_csv('../../graphs/list_of_edges_post2020.csv')

    del pd_eval
