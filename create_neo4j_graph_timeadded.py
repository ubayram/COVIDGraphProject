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
import os

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
        new_list.append(curr_date.replace('-', ''))
    return new_list

def updateScoreDate(keyitem, curr_date, data_dict):

    if keyitem not in data_dict.keys():
            data_dict[keyitem] = {}
            data_dict[keyitem]['score'] = 1 # it exists in the current abstract
            data_dict[keyitem]['date'] = [curr_date]
    else:
        data_dict[keyitem]['score'] += 1
        data_dict[keyitem]['date'].append(curr_date)
    
    return data_dict

def updateWeights(links, node_weights, tuples_weight, curr_date):

    nodes_list = []
    for item in links:
        if item[0] == item[1]:
            continue
        if item not in tuples_weight.keys():
            tuples_weight[item] = {}
            tuples_weight[item]['score'] = 1
            tuples_weight[item]['date'] = [curr_date]
        else: # it exists in the tuples_weight dictionary - from another abstract
            tuples_weight[item]['score'] += 1
            tuples_weight[item]['date'].append(curr_date)

        nodes_list.append(item[0])
        nodes_list.append(item[1])

    nodes_list = list(set(nodes_list))
    for curr_entity in nodes_list:
        if curr_entity not in node_weights.keys():
            node_weights[curr_entity] = {}
            node_weights[curr_entity]['score'] = 1
            node_weights[curr_entity]['date'] = [curr_date]
        else:
            node_weights[curr_entity]['score'] += 1
            node_weights[curr_entity]['date'].append(curr_date)

    return node_weights, tuples_weight

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

#data_rels ={}
#nodes_data = {}
node_weights_all = {}
tuples_weight_all = {}
for i in range(len(all_abstracts)):
    curr_abstract = ignoreAbbreviations(all_abstracts[i])
    curr_date = all_timestamps[i]
    links = lm.extract_link(nlp(curr_abstract))
    # get unique list of entity-relation triplets - so no repetitions
    links = list(set(links))

    node_weights_all, tuples_weight_all = updateWeights(links, node_weights_all, tuples_weight_all, curr_date)

# ------------- write the nodes to import file

fo_nodes = open('data/db_nodes.csv', 'w')
# header
fo_nodes.write("Entity:ID,Score:INT,PublishDate:INT[],:LABEL")

for curr_node in node_weights_all.keys():
    all_dates = ";".join(node_weights_all[curr_node]['date'])
    fo_nodes.write('\n' + curr_node + ',' + str(node_weights_all[curr_node]['score']) + ',"' + all_dates + '",Entity')
    
fo_nodes.close()

# ---------- write the relations to import file

# relations between the IPs - multiple connections between same IP pairs is allowed - to allow time-domain data play
f_edges = open('data/db_relationships.csv', 'w')
f_edges.write(":START_ID,Score:INT,PublishDate:INT[],:END_ID,:TYPE")

for rel_item in tuples_weight_all.keys():
    all_dates = ";".join(tuples_weight_all[rel_item]['date'])
    f_edges.write("\n" + rel_item[0] + "," + str(tuples_weight_all[rel_item]['score']) + ',"' + all_dates + '",' + rel_item[1] + "," + rel_item[2])
f_edges.close()

# now, run the terminal command to empty existing graph and create/import the data and relations in the csv files into a graph database in neo4j
os.system('sudo rm -r /var/lib/neo4j/data/databases/graph.db/')
os.system('sudo chown -R neo4j.neo4j /var/lib/neo4j/data/databases/graph.db/')
os.system('sudo neo4j-admin import --nodes "data/db_nodes.csv" --relationships "data/db_relationships.csv"')
os.system('sudo chown -R neo4j.neo4j /var/lib/neo4j/data/databases/graph.db/')