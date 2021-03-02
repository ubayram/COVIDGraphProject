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
# This is where the undirected, weighted graph construction happens
import graph_tool.all as gt
import pandas as pd
#from codetiming import Timer
import numpy as np
import networkx as nx

def getNodesForRemoval():

    fo = open('../../graphs/post2020_vertices_for_removal.txt', 'r')
    list_nodes = fo.read().split('\n')[:-1]

    return list_nodes

def checkNode(curr_node, list_nodes_remove):

    if curr_node in list_nodes_remove or len(curr_node) == 1:
        return False
    else:
        return True

def computeWeights(df, list_acceptable):

    N_valid_papers = 0
    empty_list = []
    # initialize the dataframe with the dtypes specified for efficient memory use
    new_df = pd.DataFrame({'node1' : pd.Series(empty_list, dtype='category'), 'node2': pd.Series(empty_list, dtype='category'), 
    'weight': pd.Series(empty_list, dtype='float'), 'timestamps': pd.Series(empty_list, dtype='category')})

    for i_row in range(len(df.index)):
        if i_row not in list_acceptable:
            continue
        curr_tuples = df.list_of_edges[i_row]
        curr_time = df.timestamp[i_row]

        if len(curr_tuples) >= 10: # If a paper has at least 10 links returned, including repetitions, and words are in English, it's valid
            new_df = updateWeights(new_df, curr_tuples, curr_time)
            N_valid_papers += 1
        #t.stop()
        print('Current paper processed:' + str(i_row))

    new_df = normalizeWeights(new_df, N_valid_papers)

    print('Number of valid papers used in the graph construction=' + str(N_valid_papers))
    # normalize the weights
    return new_df

def normalizeWeights(new_df, N_valid_papers):

    for i_row in range(len(new_df['node1'])):
        new_df.loc[i_row, 'weight'] = new_df.weight[i_row]/float(N_valid_papers)

    return new_df

def getValidTuplesList(list_tuples):
    list_nodes_remove = getNodesForRemoval()
    new_list_tuples = []
    for curr_tuple in list_tuples:
        is_valid1 = checkNode(curr_tuple[0], list_nodes_remove)
        is_valid2 = checkNode(curr_tuple[1], list_nodes_remove)

        #reverse_tuple = (curr_tuple[1], curr_tuple[0])
        if is_valid1 and is_valid2:
            new_list_tuples.append(curr_tuple)

    return new_list_tuples

def updateWeights(pd_eval, list_curr_tuples, curr_timestamp):

    curr_tuples = getValidTuplesList(list_curr_tuples)

    n_tuples = len(curr_tuples) # number of all connections present in the paper, doesn't have to be unique

    unique_tuples = list(set(curr_tuples))
    start_nodes = []
    end_nodes = []
    weights_ = []
    timestamps_ = []
    list_existing_tuples = list(zip(pd_eval['node1'], pd_eval['node2']))
    list_reverse_tuples = list(zip(pd_eval['node2'], pd_eval['node1']))
    
    #print(list_existing_tuples[:5])
    #print(pd_eval)
    for curr_tuple in unique_tuples:
        #reverse_tuple = (curr_tuple[1], curr_tuple[0])
        curr_count = len(np.where(np.array(curr_tuples) == curr_tuple)[0])# + len(np.where(np.array(curr_tuples) == reverse_tuple)[0])
        # check if the pair exists in the dataframe

        if curr_tuple in list_existing_tuples:
            i_row = list_existing_tuples.index(curr_tuple)#pd_eval[(pd_eval['node1'] == curr_tuple[0]) & (pd_eval['node2'] == curr_tuple[1])].index.tolist()[0]
            pd_eval.loc[i_row, 'weight'] += curr_count / (float(n_tuples))
            pd_eval.loc[i_row, 'timestamps'] += str(';' + curr_timestamp)
        elif curr_tuple in list_reverse_tuples:
            i_row = list_reverse_tuples.index(curr_tuple)#pd_eval[(pd_eval['node1'] == curr_tuple[0]) & (pd_eval['node2'] == curr_tuple[1])].index.tolist()[0]
            pd_eval.loc[i_row, 'weight'] += curr_count / (float(n_tuples))
            pd_eval.loc[i_row, 'timestamps'] += str(';' + curr_timestamp)
        else:
            start_nodes.append(curr_tuple[0])
            end_nodes.append(curr_tuple[1])
            weights_.append(curr_count / (float(n_tuples)))
            timestamps_.append(curr_timestamp)

    # append the new stuff to the dataframe
    #new_df = pd.DataFrame({'node1' : pd.Series(empty_list, dtype='category'), 'node2': pd.Series(empty_list, dtype='category'), 'weight': pd.Series(empty_list, dtype='float32')})
    pd_eval = pd_eval.append(pd.DataFrame({'node1' : pd.Series(start_nodes, dtype='category'), 'node2': pd.Series(end_nodes, dtype='category'), 
    'weight': pd.Series(weights_, dtype='float'), 'timestamps': timestamps_}), ignore_index=True)

    return pd_eval

def createGraph(pd_eval, savename, select_):

    if 'gt' in select_:
        g = gt.Graph(directed=False)
        # Set property maps for edge attributes
        weight = g.new_edge_property('float')

        # Add edges and nodes in bulk
        node_id = g.add_edge_list(pd_eval.values, hashed=True, eprops=[weight]) 
        g.vertex_properties['node_id'] = node_id
        g.edge_properties['weight'] = weight

        # Save graph
        g.save(savename + '.graphml')
    else:
        # create a networkX version
        g2 = nx.from_pandas_edgelist(pd_eval, source='node1', target='node2', edge_attr='weight', create_using=nx.Graph())
        nx.write_gml(g2, savename + '_netx.gml.gz')
        del g2
    
        pd_eval.to_csv(savename + '.csv', index=False)


def eliminateRows(df_links):

    n = len(df_links.index)
    list_acceptable = []
    for i_row in range(n):
        curr_date = df_links.timestamp[i_row]
        year_ = int(curr_date.split('-')[0])

        if year_ >= 2020:
            list_acceptable.append(i_row)

    return list_acceptable

def completeGraphConstruction(select_):

    print('Graph post 2020')
    # list of rows where years are before and after 2020
    df_links = pd.read_csv('../../graphs/list_of_edges_post2020.csv', converters={'list_of_edges': eval})
    list_acceptable = eliminateRows(df_links)

    pd_eval = computeWeights(df_links, list_acceptable)
    createGraph(pd_eval, '../../graphs/graph_postCOVID_final', select_)
    del pd_eval
