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
# This code calls the necessary inputs (graph) and the functions to split the graph into training/validation/test parts
# and performs the selected link prediction approach, and saves the results
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
