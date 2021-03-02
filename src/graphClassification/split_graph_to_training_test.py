#Author: Ulya Bayram
#email : ulya.bayram@comu.edu.tr
#
# The original versions of these scripts can be found from the StellarGraph website
#------------------------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------------------------
#
#These code are writen for a research project, published in OIR. If you use any of them, please cite:

#Ulya Bayram, Runia Roy, Aqil Assalil, Lamia Ben Hiba, 
#"The Unknown Knowns: A Graph-Based Approach for Temporal COVID-19 Literature Mining", 
#Online Information Review (OIR), COVID-19 Special Issue, 2021.
#
#------------------------------------------------------------------------------------------------------
# Code to split the graph into training and test parts for running node2vec, and other link prediction methods
import pandas as pd
#from codetiming import Timer
import numpy as np
import networkx as nx
import stellargraph as sg
from math import isclose
import os
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from collections import Counter
from sklearn.model_selection import train_test_split

def splitSampleGraph():

    print('Graph post 2020')
    graph = nx.read_gml('../../graphs/graph_postCOVID_final_netx.gml.gz')
    
    for i in range(1, 6):
        print('Current run ' + str(i))

        # Define an edge splitter on the original graph:
        edge_splitter_ = EdgeSplitter(graph)

        # Randomly sample a fraction p of the graph (positive links), and same number of negative links, from graph, and obtain the
        # reduced graph graph_subset with the sampled links removed:
        graph_, sampled_edges, sample_labels = edge_splitter_.train_test_split(p=0.5, method="global")

        nx.write_gml(graph_, '../../graphs/graph_sampled_' + str(i) + '.gml.gz')

        del graph_

        # Now, split the sampled edges into training-test-validation sets for performing link prediction

        # Split operation 1 - obtain test versus train+validation
        (sampled_comp, sampled_test, labels_comp, labels_test,) = train_test_split(sampled_edges, sample_labels, train_size=0.65, test_size=0.35)

        # Split operation 2 - divide the comp block into training and validation sets
        (sampled_training, sampled_validation, labels_training, labels_validation,) = train_test_split(sampled_comp, labels_comp, train_size=0.77, test_size=0.23)

        # Save the sampled training validation test sets
        df_train = pd.DataFrame({'node1': np.array(sampled_training)[:, 0], 'node2': np.array(sampled_training)[:, 1], 'labels': labels_training})
        df_train.to_csv('../../graphs/graph_train_edges_sampled_' + str(i) + '.csv')
        del df_train

        print('Number of training samples (positive) ' + str(len(labels_training)/2.0))

        df_val = pd.DataFrame({'node1': np.array(sampled_validation)[:, 0], 'node2': np.array(sampled_validation)[:, 1], 'labels': labels_validation})
        df_val.to_csv('../../graphs/graph_val_edges_sampled_' + str(i) + '.csv')
        del df_val

        print('Number of validation samples (positive) ' + str(len(labels_validation)/2.0))

        df_test = pd.DataFrame({'node1': np.array(sampled_test)[:, 0], 'node2': np.array(sampled_test)[:, 1], 'labels': labels_test})
        df_test.to_csv('../../graphs/graph_test_edges_sampled_' + str(i) + '.csv')
        del df_test

        print('Number of test samples (positive) ' + str(len(labels_test)/2.0))
