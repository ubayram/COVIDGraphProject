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
# This code calls the previously cleaned texts and extracts necessary nodes and edges, and computes connection weights
# and creates a graph
import os
import argparse
import sys
from src.graphConstruction import extract_nodes_relations_from_texts as enr
from src.graphConstruction import create_undirected_graph as cug

if __name__ == '__main__':

    # First, call the script that reads the processed text files and 
    # extracts nodes and edges, and computes connection weights from each texts to construct a graph

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--savedir", 
        default="./", 
        type=str, 
        required=False,
        help="Give the directory you previously used to save the resulting csv file full_cord19_texts.csv.\n Default is the current directory."
    )
    parser.add_argument(
        "--selection", 
        default='netx', 
        type=str, 
        required=False,
        help="Select what type of graph you want to save: netx for NetworkX, gt for Graph tool.\n Default is netx."
    )
    args = parser.parse_args()

    input_dir = args.savedir
    select_ = args.selection

    enr.extract_nodes_edges(input_dir + 'full_cord19_texts.csv')
    cug.completeGraphConstruction(select_)
