# Author: Ulya Bayram
# Contact: ulyabayram@gmail.com
# This code calls essential data cleaning, filling, validating scripts over the CORD-19 data and metadata
# Developed to perform on the 2020-05-19 version of the dataset
import os
import argparse
import sys
from src.preprocessCORD19 import extract_improved_metadata as ex
from src.preprocessCORD19 import fetch_date_api as fetch_
from src.preprocessCORD19 import fix_times_preprocess_texts as fix_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", 
        default="document_parses/*/", 
        type=str, 
        required=False,
        help="Give the full directory for the json files, use * to cover all included folder names. Default is: document_parses/*/"
    )
    parser.add_argument(
        "--metadatadir", 
        default="./", 
        type=str, 
        required=False,
        help="Give the directory for the metadata.csv file. Default is the current directory."
    )
    parser.add_argument(
        "--savedir", 
        default="./", 
        type=str, 
        required=False,
        help="Directory to save the resulting csv files. Default is the current directory."
    )
    args = parser.parse_args()

    datadir = args.datadir
    savedir = args.savedir
    metadatadir = args.metadatadir

    print('A new folder called processed_texts is created in savedir that contains full text bodies stripped from the json files')
    os.mkdir(savedir + 'processed_texts')
    new_metadata_df, bad_times_df = ex.extractNewImprovedMetadata(datadir, savedir, metadatadir)
    corrected_times_df = fetch_.fetchDate(bad_times_df, email="email", column_name='doi')
    fix_.fixCollection(new_metadata_df, corrected_times_df, savedir)