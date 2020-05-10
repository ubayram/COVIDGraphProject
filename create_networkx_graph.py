# Ulya Bayram
# ulyabayram@gmail.com
#

import subject_verb_object_extract_LB as lm # get Lamia's code
import networkx as nx
from collections import Counter
import spacy
import en_core_web_sm
import pandas as pd
import numpy as np
from collections import Counter
import datetime as dt
import nltk
import regex as re

def get_sentences(curr_abstract):

    curr_abstract = ignoreAbbreviations(curr_abstract)
    curr_abstract = separateNumberPeriod(curr_abstract)
    sentences = nltk.tokenize.sent_tokenize(curr_abstract)
    #cont = [sen_ for sen_ in sentences[1:]  if (sen_ != '')&(~sen_.isnumeric())]
    return sentences

def ignoreAbbreviations(curr_abstract):
    #curr_abstract = curr_abstract.replace('≥', ' ')
    #curr_abstract = curr_abstract.replace('  ', ' ')
    list_abbreviations = ['i.e.', 'e.g.', 'e.g', 'v.s.', 'inc.', 'etc.']
    clean_abbreviations = ['ie', 'eg', 'eg', 'vs', 'inc', 'etc']
    list_tokens = curr_abstract.split(' ')

    for abbr in list_abbreviations:
        if abbr in list_tokens:
            i_change = list_tokens.index(abbr)
            list_tokens[i_change] = clean_abbreviations[list_abbreviations.index(abbr)]

    return ' '.join(list_tokens)

def separateNumberPeriod(curr_abstract):
    return re.sub(r'(?<=\d)\. ', ' . ', curr_abstract)

def convertDatesToNums(timelist):

    new_list = []
    for curr_date in timelist:
        new_list.append(int(curr_date.replace('-', '')))
    return new_list

# weights will increase by 1 here for each node and each relation if they exist
def updateWeightsTime(links, node_weights, tuples_weight, time_key):

    nodes_list = []
    for item in links:
        if item not in tuples_weight[time_key].keys():
            tuples_weight[time_key][item] = 1
        else:
            tuples_weight[time_key][item] += 1
        nodes_list.append(item[0])
        nodes_list.append(item[1])

    nodes_list = list(set(nodes_list))
    for curr_entity in nodes_list:
        if curr_entity not in node_weights[time_key].keys():
            node_weights[time_key][curr_entity] = 1
        else:
            node_weights[time_key][curr_entity] += 1

    return node_weights, tuples_weight

def updateWeights(links, node_weights, tuples_weight):

    nodes_list = []
    for item in links:
        if item not in tuples_weight.keys():
            tuples_weight[item] = 1
        else:
            tuples_weight[item] += 1
        nodes_list.append(item[0])
        nodes_list.append(item[1])

    nodes_list = list(set(nodes_list))
    for curr_entity in nodes_list:
        if curr_entity not in node_weights.keys():
            node_weights[curr_entity] = 1
        else:
            node_weights[curr_entity] += 1

    return node_weights, tuples_weight

def initializeTDomainDict(dict_):

    dict_['pre'] = {}
    dict_['jan'] = {}
    dict_['feb'] = {}
    dict_['mar'] = {}
    dict_['apr'] = {}

    return dict_

def getTDomainKey(curr_time):

    first_splitter = 20200101
    jan_end = 20200131
    feb_end = 20200229
    mar_end = 20200331

    if curr_time < first_splitter:
        return 'pre'
    elif curr_time >= first_splitter and curr_time <= jan_end:
        return 'jan'
    elif curr_time > jan_end and curr_time <= feb_end:
        return 'feb'
    elif curr_time > feb_end and curr_time <= mar_end:
        return 'mar'
    elif curr_time > mar_end:
        return 'apr'
    else:
        return 'error'

# ----------------------- main ----------------------
# use spacy small model
nlp = en_core_web_sm.load()

path_to_stopwords = 'nltk_stopwords.txt'
    
# Keep this block
with open(path_to_stopwords, 'r') as file:
    stopwords = file.readlines()
for word in stopwords:
    nlp.vocab[word].is_stop = True

# collect the abstracts
all_data = pd.read_csv('processed_dataset.csv')
all_abstracts = list(all_data.abstract)
all_timestamps = convertDatesToNums(list(all_data.publish_time))

node_weights = {}
tuples_weight = {}
node_weights = initializeTDomainDict(node_weights)
tuples_weight = initializeTDomainDict(tuples_weight)

node_weights_all = {}
tuples_weight_all = {}
for i in range(len(all_abstracts)):
    curr_abstract = ignoreAbbreviations(all_abstracts[i])
    curr_time = all_timestamps[i]
    time_key = getTDomainKey(curr_time)
    #print(curr_abstract)
    links = lm.extract_link(nlp(curr_abstract))
    # get unique list of entity-relation triplets - so no repetitions
    links = list(set(links))
    print(i)

    node_weights, tuples_weight = updateWeightsTime(links, node_weights, tuples_weight, time_key)
    node_weights_all, tuples_weight_all = updateWeights(links, node_weights_all, tuples_weight_all)

# Separate for time domain
for k in node_weights.keys():
    source = []
    target = []
    relations = []
    weights = []
    save_filename = 'knowledge_graph_time_' + k + '.gml.gz'
    for curr_tuple in tuples_weight[k].keys():
        source.append(curr_tuple[0])
        target.append(curr_tuple[1])
        relations.append(curr_tuple[2])
        weights.append(tuples_weight[k][curr_tuple])

    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations, 'weight':weights})
    G=nx.from_pandas_edgelist(kg_df,"source", "target", edge_attr=True, create_using=nx.DiGraph())

    nx.set_node_attributes(G, node_weights[k], 'weight')

    nx.write_gml(G, save_filename)
    print('Current time for the following graph ' + k)
    print(G.number_of_nodes())
    print(G.number_of_edges())

    del G

# Complete graph
source = []
target = []
relations = []
weights = []
save_filename = 'knowledge_graph_complete.gml.gz'
for curr_tuple in tuples_weight_all.keys():
    source.append(curr_tuple[0])
    target.append(curr_tuple[1])
    relations.append(curr_tuple[2])
    weights.append(tuples_weight_all[curr_tuple])

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations, 'weight':weights})
G=nx.from_pandas_edgelist(kg_df,"source", "target", edge_attr=True, create_using=nx.DiGraph())

nx.set_node_attributes(G, node_weights_all, 'weight')

nx.write_gml(G, save_filename)
print('All')
print(G.number_of_nodes())
print(G.number_of_edges())