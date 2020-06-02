# Lamia Ben hiba
# lamia.benhiba@gmail.com
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


def update_dataframe(data):
    '''
    :param data: the dataframe that we want to update with the dates
    :return: Nothing
    '''
	data['date_API'] = ''
	for ix, row in data.iterrows():
		data.loc[ix, 'date_API'] = fetch_all_dates(str(row['doi']))


def main(filepath, email, targetpath):
    '''
    Based on a csv file, we create a dataframe, then we populate the date using the pubmed API
    :param filepath: path of csv file that will serve as source for our dataframe
    :param email: the email to use for API requests [Entrez requirement]
    :return: Nothing
    '''
	data = pd.read_csv(filepath)
	Entrez.email = email
	update_dataframe(data)
	data.to_excel(targetpath)

if __name__ == '__main__':
	email = "email_here"
	filepath = './papers_to_scrape_for_timestamps.csv'
	targetpath = './updated_papers_timestamps.csv'
	main(filepath,email, targetpath)
    
