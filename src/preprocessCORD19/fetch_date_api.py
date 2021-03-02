#Author: Lamia Ben hiba
#email : lamia.benhiba@gmail.com
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
# Enriching the metadata with missing dates

import pandas as pd
from Bio import Entrez
from Bio import Medline
from io import StringIO

def search_pubmed(doi):
    '''
    :param doi: DOI of paper to use for searching
    :return: Python dict with API search results
    '''
    search_term = doi+"[AID]"
    search = Entrez.esearch(db="pubmed", term = search_term)
    handle = Entrez.read(search)
    try:
        return handle
    except Exception as e:
        raise IOError(str(e))
    finally:
        search.close()

def fetch_paper_date(record_id):
    '''
    :param record_id: id of the record we'd like to fetch in its entirety from the database
    :return: the record
    '''
    search = Entrez.efetch(db="pubmed", id=record_id,rettype="Medline",retmode="text")
    rec = search.read()
    return rec


def fetch_all_dates(doi):
    '''
    :param doi: DOI of paper to use for searching
    :return: the date of publication when applicable
    '''
    handle = search_pubmed(doi)
    if len(handle['IdList'])>0:
        record_id = handle['IdList'][0]
        paper = fetch_paper_date(record_id)
        rec_file = StringIO(paper)
        medline_rec = Medline.read(rec_file)
        return medline_rec['PHST'][-1].partition(' ')[0]
    else:
        return None


def update_dataframe(data, column_name = 'doi'):
    '''
    :param data: the dataframe that we want to update with the dates
    :param column_name: the name of data column to use for searching in the API, default doi
    :return: Nothing
    '''
    data['date_API'] = ''
    for ix, row in data.iterrows():
        if row[column_name]:
            data.loc[ix, 'date_API'] = fetch_all_dates(str(row[column_name]))
        else:
            data.loc[ix, 'date_API'] = None
        #print(ix)
    return data


def fetchDate(data, email="email", column_name='doi'):
    '''
    Based on a csv file, we create a dataframe, then we populate the date using the pubmed API.
    :param filepath: path of csv file that will serve as source for our dataframe
    :param email: the email to use for API requests [Entrez requirement]
    :param column_name: the name of data column to use for searching in the API, default doi
    :return: file locally created with the right timestamps
    '''
    #data = pd.read_csv(filepath)
    Entrez.email = email
    return update_dataframe(data, column_name) #.to_csv('data_with_timestamps.csv')

'''
if __name__ == '__main__':
    email = "email"
    filepath = 'file_path'
    column_name = 'doi' #column name for search
    main(filepath,email, column_name)
    
'''
