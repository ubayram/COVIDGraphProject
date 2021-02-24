# Author: Ulya Bayram
# Contact: ulyabayram@gmail.com
#
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
