# Ulya Bayram
# ulyabayram@gmail.com
#

import networkx as nx
from collections import Counter
import spacy
import en_core_web_sm
import pandas as pd
import numpy as np

def saveResults(results, savename):
    
    fo = open(savename, 'w')
    
    for node_s in results:
        fo.write(node_s[0] + '\t' + str(node_s[1]) + '\n')

    fo.close()

# ----------------------- main ----------------------
time_list = ['pre', 'jan', 'feb', 'mar', 'apr']
'''
# Separate KG for each time domain
for k in time_list:

    filename = 'graphs/knowledge_graph_time_' + k + '.gml.gz'
    G = nx.read_gml(filename)

    # simple degrees
    unweighted_degrees = G.degree()
    saveResults(unweighted_degrees, 'graph_analysis/kg_' + k + '_unweighted_degrees.txt')

    weighted_degrees = G.degree(weight='weight')
    saveResults(weighted_degrees, 'graph_analysis/kg_' + k + '_weighted_degrees.txt')

    # in and out degrees
    unweighted_degrees = G.in_degree()
    saveResults(unweighted_degrees, 'graph_analysis/kg_' + k + '_unweighted_in_degrees.txt')

    weighted_degrees = G.in_degree(weight='weight')
    saveResults(weighted_degrees, 'graph_analysis/kg_' + k + '_weighted_in_degrees.txt')

    unweighted_degrees = G.out_degree()
    saveResults(unweighted_degrees, 'graph_analysis/kg_' + k + '_unweighted_out_degrees.txt')

    weighted_degrees = G.out_degree(weight='weight')
    saveResults(weighted_degrees, 'graph_analysis/kg_' + k + '_weighted_out_degrees.txt')

    # closeness centrality https://toreopsahl.com/2010/03/20/closeness-centrality-in-networks-with-disconnected-components/
    
    del G
'''

filename = 'graphs/knowledge_graph_complete.gml.gz'
G = nx.read_gml(filename)

# simple degrees
unweighted_degrees = G.degree()
saveResults(unweighted_degrees, 'graph_analysis/kg_global_unweighted_degrees.txt')

weighted_degrees = G.degree(weight='weight')
saveResults(weighted_degrees, 'graph_analysis/kg_global_weighted_degrees.txt')

# in and out degrees
unweighted_degrees = G.in_degree()
saveResults(unweighted_degrees, 'graph_analysis/kg_global_unweighted_in_degrees.txt')

weighted_degrees = G.in_degree(weight='weight')
saveResults(weighted_degrees, 'graph_analysis/kg_global_weighted_in_degrees.txt')

unweighted_degrees = G.out_degree()
saveResults(unweighted_degrees, 'graph_analysis/kg_global_unweighted_out_degrees.txt')

weighted_degrees = G.out_degree(weight='weight')
saveResults(weighted_degrees, 'graph_analysis/kg_global_weighted_out_degrees.txt')