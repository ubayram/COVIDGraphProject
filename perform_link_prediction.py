# Author: Ulya Bayram
# Contact: ulyabayram@gmail.com
#
import os
import argparse
import sys
from src.graphClassification import split_graph_to_training_test as sp
from src.graphClassification import simple_link_prediction as simp
from src.graphClassification import node2vec_link_prediction as n2v

if __name__ == '__main__':

    # First, call the script that reads the processed text files and 
    # extracts nodes and edges, and computes connection weights from each texts to construct a graph

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--savedir", 
        default="./", 
        type=str, 
        required=False,
        help="Give the directory to save the classification results.\n Default is the current directory."
    )
    parser.add_argument(
        "--method", 
        default='adar', 
        type=str, 
        required=False,
        help="Select your method: node2vec, adar, or jaccard.\n Default is adar."
    )
    parser.add_argument(
        "--classifier", 
        default='naiveBayes', 
        type=str, 
        required=False,
        help="Select your classifier: naiveBayes, logisticRegression, or deepLearning.\n Default is naiveBayes."
    )
    args = parser.parse_args()

    save_dir = args.savedir
    method_ = args.method
    classifier_ = args.classifier

    # First split the graph into training and test (repeats it randomly 5 times for Monte Carlo cross-validation)
    # saves the samples to 'graphs' folder
    sp.splitSampleGraph()

    if 'node2vec' in method_:
        # Code is only for split 5, with a simple for loop you can modify it to perform on all splits
        # Keep in mind that each split takes a very long time so you probably will need a server
        if 'logisticRegression' in classifier_:
            n2v.applyLogisticRegression(save_dir)
        elif 'deepLearning' in classifier_:
            n2v.applyDeepLearning(save_dir)
        else:
            n2v.applyNaiveBayes(save_dir)
    else:

        if 'logisticRegression' in classifier_:
            simp.applyLogisticRegression(save_dir, method_)
        elif 'deepLearning' in classifier_:
            simp.applyDeepLearning(save_dir, method_)
        else:
            simp.applyNaiveBayes(save_dir, method_)