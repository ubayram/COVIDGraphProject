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
