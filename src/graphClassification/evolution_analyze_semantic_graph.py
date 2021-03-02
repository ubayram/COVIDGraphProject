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
# This code performs simple temporal analysis for detecting the evolution of selected concepts
# in the progress of COVID-19 pandemic
import graph_tool.all as gt
import pandas as pd
import numpy as np
from collections import Counter

# select 06-14 as the last date
def getGrowthPerNode(df):

    nodes_dict = {}

    for i_row in range(len(df.node1)):

        node1 = df.node1[i_row]
        node2 = df.node2[i_row]
        list_dates = df.timestamps[i_row].split(';')

        if node1 in nodes_dict.keys():
            nodes_dict[node1]+= list_dates
        else:
            nodes_dict[node1] = list_dates

        if node2 in nodes_dict.keys():
            nodes_dict[node2]+= list_dates
        else:
            nodes_dict[node2] = list_dates

    return nodes_dict

def sortDates(nodes_dict):
    
    list_nodes = []
    list_nums = []
    for c_node in nodes_dict.keys():
        nodes_dict[c_node] = sorted(nodes_dict[c_node]) # will help plotting them. Also compute speed
        list_nodes.append(c_node)
        list_nums.append(len(nodes_dict[c_node]))

    print(max(list_nums))
    list_nodes = np.array(list_nodes)
    indices_list = [i[0] for i in sorted(enumerate(list_nums), key=lambda x:x[1], reverse=True)]
    print(list_nodes[indices_list[:100]])

def saveNodevsDate(nodes_dict, save_dir):
    list_nodes = ['phenotype', 'hpaiv', 'pneumonia', 'viral spread', 'uncertainty', 'stata version', 'outbreak', 'disease propagation',
        'lung epithelia', 'cytopathogenic effect', 'furin-like cleavage site', 'super-infection', 'protein', 'exposure to']
    #list_dates = []
    for c_node in list_nodes:
        #list_dates.append(nodes_dict[c_node])
        filename = c_node.replace(' ', '_')

        fo = open(save_dir+'node_' + filename + '.txt', 'w')
        fo.write('date\n')

        list_dates = sorted(nodes_dict[c_node])

        for c_date in list_dates:
            fo.write(c_date + '\n')

    #df = pd.DataFrame({'node': list_nodes, 'date':list_dates})
    #df.to_csv('post2020_dates_per_selected_node.csv')

def saveNodesDateOccurrences(nodes_dict, save_dir):

    fo = open('relevant_nodes.txt', 'r')
    list_nodes = fo.read().split('\n')[:-1]
    fo.close()
    for c_node in list_nodes:
        list_dates = nodes_dict[c_node]

        # get the count of each date
        dates_dict = Counter(list_dates)

        filename = c_node.replace(' ', '_')
        fo_ = open(save_dir + '/' + filename + '.txt', 'w')
        fo_.write('date,count,accum\n')
        acc = 0

        list_unique_dates = sorted(dates_dict.keys())
        for c_date in list_unique_dates:
            acc += dates_dict[c_date]
            fo_.write(c_date + ',' + str(dates_dict[c_date]) + ',' + str(acc) + '\n')
        fo_.close()

def saveMonthlyGrowth(df):
    num_nodes_by_date = {}
    num_edges_by_date = {}
    for i_row in range(len(df.node1)):

        node1 = df.node1[i_row]
        node2 = df.node2[i_row]
        list_dates = df.timestamps[i_row].split(';')

        for c_date in list_dates:
            if c_date in num_edges_by_date.keys():
                num_edges_by_date[c_date] += 1
                num_nodes_by_date[c_date].append(node1)
                num_nodes_by_date[c_date].append(node2)
            else:
                num_edges_by_date[c_date] = 1
                num_nodes_by_date[c_date] = [node1, node2]

    return num_edges_by_date, num_nodes_by_date

def saveGrowthOfNodesEdgesPerDay(df, save_dir):

    num_edges_by_date, num_nodes_by_date = saveMonthlyGrowth(df)

    list_dates = sorted(num_edges_by_date.keys())

    # num of new nodes and edges per day
    fnew = open(save_dir + 'list_new_data_daily.txt', 'w')
    fnew.write('date,numedges,numnodes\n')
    # accumulative num of nodes and edges per day
    facc = open(save_dir + 'list_accumulative_data_daily.txt', 'w')
    facc.write('date,numedges,numnodes\n')

    list_nodes = []
    edge_accum = 0
    for c_date in list_dates:

        list_curr_nodes = list(set(num_nodes_by_date[c_date]))
        new_node_counter = 0
        for c_node in list_curr_nodes:
            if c_node not in list_nodes:
                new_node_counter += 1
        fnew.write(c_date + ',' + str(num_edges_by_date[c_date]) + ',' + str(new_node_counter) + '\n')

        list_nodes += list_curr_nodes
        list_nodes = list(set(list_nodes))

        edge_accum += num_edges_by_date[c_date]
        facc.write(c_date + ',' + str(edge_accum) + ',' + str(len(list_nodes)) + '\n')

    facc.close()
    fnew.close()

def mergeByHalfMonthsOnlyNew(save_dir):
    fnew = open(save_dir + 'list_new_data_daily.txt', 'r')
    fnew2 = open(save_dir + 'list_new_data_biweekly.txt', 'w')
    fnew2.write('date,numedges,numnodes\n')
    i = 0
    for line in fnew:
        if i == 0:
            i = 1
            continue
        c_date = line.split(',')[0]

        day_ = int(c_date.split('-')[2])

        if day_ == 1:
            numedges = 0
            numnodes = 0
        
        if day_ < 15:
            numedges += int(line.split(',')[1])
            numnodes += int(line.split(',')[2])

        if day_ == 15:
            fnew2.write(c_date + ',' + str(numedges) + ',' + str(numnodes) + '\n')
            numedges = 0
            numnodes = 0

        if day_ > 15:
            numedges += int(line.split(',')[1])
            numnodes += int(line.split(',')[2])

        if int(c_date.split('-')[1]) == 1 or int(c_date.split('-')[1]) == 3 or int(c_date.split('-')[1]) == 5:
            if day_ == 31:
                fnew2.write(c_date + ',' + str(numedges) + ',' + str(numnodes) + '\n')
                numedges = 0
                numnodes = 0
        elif int(c_date.split('-')[1]) == 2:
            if day_ == 29:
                fnew2.write(c_date + ',' + str(numedges) + ',' + str(numnodes) + '\n')
                numedges = 0
                numnodes = 0
        elif int(c_date.split('-')[1]) == 4:
            if day_ == 30:
                fnew2.write(c_date + ',' + str(numedges) + ',' + str(numnodes) + '\n')
                numedges = 0
                numnodes = 0
        elif int(c_date.split('-')[1]) == 6:
            if day_ == 11:
                fnew2.write(c_date + ',' + str(numedges) + ',' + str(numnodes) + '\n')
                numedges = 0
                numnodes = 0

    fnew2.close()
    fnew.close()

def mergeByHalfMonthsAccum(save_dir):
    fnew = open(save_dir + 'list_accumulative_data_daily.txt', 'r')
    fnew2 = open(save_dir + 'list_accumulated_data_biweekly.txt', 'w')
    fnew2.write('date,numedges,numnodes\n')
    i = 0
    numedges = 0
    numnodes = 0
    for line in fnew:
        if i == 0:
            i = 1
            continue
        c_date = line.split(',')[0]

        day_ = int(c_date.split('-')[2])
        numedges += int(line.split(',')[1])
        numnodes += int(line.split(',')[2])

        if day_ == 15:
            fnew2.write(c_date + ',' + str(numedges) + ',' + str(numnodes) + '\n')

        if int(c_date.split('-')[1]) == 1 or int(c_date.split('-')[1]) == 3 or int(c_date.split('-')[1]) == 5:
            if day_ == 31:
                fnew2.write(c_date + ',' + str(numedges) + ',' + str(numnodes) + '\n')
        elif int(c_date.split('-')[1]) == 2:
            if day_ == 29:
                fnew2.write(c_date + ',' + str(numedges) + ',' + str(numnodes) + '\n')
        elif int(c_date.split('-')[1]) == 4:
            if day_ == 30:
                fnew2.write(c_date + ',' + str(numedges) + ',' + str(numnodes) + '\n')
        elif int(c_date.split('-')[1]) == 6:
            if day_ == 11:
                fnew2.write(c_date + ',' + str(numedges) + ',' + str(numnodes) + '\n')

    fnew2.close()
    fnew.close()

def eliminateRows(df_links):

    n = len(df_links.index)
    list_acceptable = []
    for i_row in range(n):
        curr_date = df_links.timestamp[i_row]
        year_ = int(curr_date.split('-')[0])

        if year_ >= 2020:
            list_acceptable.append(i_row)

    return list_acceptable

def savePaperDateHist(save_dir):
    df = pd.read_csv('../../graphs/list_of_edges_post2020.csv', converters={'list_of_edges': eval})
    list_acceptable = eliminateRows(df)

    #dict_ = {}
    list_dates = []
    for i_row in range(len(df.index)):
        if i_row not in list_acceptable:
            continue
        curr_tuples = df.list_of_edges[i_row]
        curr_time = df.timestamp[i_row]

        if len(curr_tuples) >= 10:
            #if curr_time in dict_.keys():
            #    dict_[curr_time] += 1 # add the paper to the pool
            #else:
            #    dict_[curr_time] = 1 # start the paper counter
            list_dates.append(curr_time)
    #list_dates = sorted(dict_.keys())
    #list_counts = []
    #for i_date in list_dates:
    #    list_counts.append(dict_[i_date])

    ndf = pd.DataFrame({'date':list_dates})#, 'count':list_counts})
    ndf.to_csv(save_dir + 'dataset_papers_histogram.csv')

# selected popular nodes - interesting ones: helsinki declaration, phenotype, chinese team, superfamily, pneumonia, virulence
# statistical significance, health care system, parameter fitting procedure, molecular analysis, lung epithelia
# strain-specific polymerase chain reaction analysis, super-infection
def runAllAnalysis(save_dir):

    print('Graph post 2020')

    df = pd.read_csv('../../graphs/graph_postCOVID_final.csv')
    savePaperDateHist(save_dir)
    nodes_dict = getGrowthPerNode(df)
    sortDates(nodes_dict)
    saveNodevsDate(nodes_dict, save_dir)
    #saveNodesDateOccurrences(nodes_dict, save_dir)
    saveMonthlyGrowth(nodes_dict)
    mergeByHalfMonthsOnlyNew(save_dir)
    mergeByHalfMonthsAccum(save_dir)
