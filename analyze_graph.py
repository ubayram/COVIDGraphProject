# Author: Ulya Bayram
# Contact: ulyabayram@gmail.com
#
import os
import argparse
import sys
from src.graphClassification import evolution_analyze_semantic_graph as eva

if __name__ == '__main__':

    # First, call the script that reads the processed text files and 
    # extracts nodes and edges, and computes connection weights from each texts to construct a graph

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--savedir", 
        default="./", 
        type=str, 
        required=False,
        help="Give the directory you want to save the evolution analysis results to.\n Default is the current directory."
    )
    args = parser.parse_args()

    save_dir = args.savedir

    eva.runAllAnalysis(save_dir)

    print('All analysis-worthy data extracted from the graph and saved to your preferred directory.')
    print('Scripts for plotting them are not included here.')
