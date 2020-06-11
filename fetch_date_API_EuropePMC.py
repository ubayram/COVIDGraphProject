# Author: Lamia ben hiba
# Contact: lamia.benhiba@gmail.com
# This code corrects erroneous dates by fetching the correct ones from Europe PMC API

import pandas as pd
import requests
import argparse

def read_file(file_path):
	data = pd.read_csv(filepath)
	data['date_API'] = ''
	return data

def fetch_date(data):
	for ix, row in data.iterrows():
    	if row['date_API'] == '':
        	print(ix)
        	doi = str(row['doi'])
        	url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query='+doi
        	response = requests.get(url)
        	try:
            	data_new.loc[ix, 'date_API'] = str(response.content).split('<firstPublicationDate>')[1].split('<')[0]
        	except IndexError:
        		print(doi, ' not found')
            	pass
    return data


def main(filepath, outputpath):
	data = read_file(filepath)
	data_with_dates = fetch_date(data) 
	data_with_dates.to_csv(outputpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile", 
        default="data/papers_to_scrape_for_timestamps_with_dates_27052020.csv", 
        type=str, 
        required=True,
        help="Full path and name of the input file containing data to complete. Default is: data/papers_to_scrape_for_timestamps.csv"
    )
    parser.add_argument(
        "--outputfile", 
        default="./papers_updated.csv", 
        type=str, 
        required=True,
        help="Full path and name of the output file. Default is papers_updated.csv in the current directory."
    )

    inputfile = args.inputfile
    outputfile = args.outputfile

    print('A new file is create in the outputfile path/name with data updated with dates from Europe PMC')
    main(inputfile, outputfile)
