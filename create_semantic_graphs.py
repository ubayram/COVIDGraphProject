# Ulya Bayram
# ulyabayram@gmail.com
#
import langdetect
import graph_tool.all as gt
import pandas as pd
#from codetiming import Timer
import numpy as np

#t = Timer()

def isEnglish(list_tuples):

    try:
        lang_ = langdetect.detect(' '.join("(%s,%s)" % tup for tup in  list_tuples))
    except langdetect.lang_detect_exception.LangDetectException:
        return False
    else:
        if  lang_ == 'en':
            return True
        else:
            return False

def computeWeights(df):

    N_valid_papers = 0
    empty_list = []
    # initialize the dataframe with the dtypes specified for efficient memory use
    new_df = pd.DataFrame({'node1' : pd.Series(empty_list, dtype='category'), 'node2': pd.Series(empty_list, dtype='category'), 'weight': pd.Series(empty_list, dtype='float')})

    for i_row in range(len(df.index)):
        curr_tuples = eliminateNumericEntities(df.list_of_edges[i_row])
        #t.start()
        is_EN = isEnglish(curr_tuples)
        #t.stop()
        #t.start()
        if len(curr_tuples) >= 10 and is_EN: # If a paper has at least 10 links returned, including repetitions, and words are in English, it's valid
            new_df = updateWeights(new_df, curr_tuples)
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

def checkEntityForNumeric(curr_entity):
    curr_entity = curr_entity.replace(' ', '')
    curr_entity = curr_entity.replace('%', '')
    curr_entity = curr_entity.replace(',', '')
    curr_entity = curr_entity.replace('.', '')

    return curr_entity.isnumeric()

def eliminateNumericEntities(list_tuples):

    new_list = []
    for c_tuple in list_tuples:
        if (not checkEntityForNumeric(c_tuple[0]) ) and (not checkEntityForNumeric(c_tuple[1])):
            new_list.append(c_tuple)

    return new_list

def updateWeights(pd_eval, curr_tuples):

    n_tuples = len(curr_tuples) # number of all connections present in the paper

    unique_tuples = list(set(curr_tuples))
    start_nodes = []
    end_nodes = []
    weights_ = []

    list_existing_tuples = list(zip(pd_eval['node1'], pd_eval['node2']))
    #print(list_existing_tuples[:5])
    #print(pd_eval)
    for curr_tuple in unique_tuples:
        curr_count = len(np.where(np.array(curr_tuples) == curr_tuple)[0])
        # check if the pair exists in the dataframe

        if curr_tuple in list_existing_tuples:
            i_row = list_existing_tuples.index(curr_tuple)#pd_eval[(pd_eval['node1'] == curr_tuple[0]) & (pd_eval['node2'] == curr_tuple[1])].index.tolist()[0]
            pd_eval.loc[i_row, 'weight'] += curr_count / (float(n_tuples))
        else:
            start_nodes.append(curr_tuple[0])
            end_nodes.append(curr_tuple[1])
            weights_.append(curr_count / (float(n_tuples)))

    # append the new stuff to the dataframe
    #new_df = pd.DataFrame({'node1' : pd.Series(empty_list, dtype='category'), 'node2': pd.Series(empty_list, dtype='category'), 'weight': pd.Series(empty_list, dtype='float32')})
    pd_eval = pd_eval.append(pd.DataFrame({'node1' : pd.Series(start_nodes, dtype='category'), 'node2': pd.Series(end_nodes, dtype='category'), 'weight': pd.Series(weights_, dtype='float')}), ignore_index=True)

    return pd_eval

def createGraph(pd_eval, savename):
    pd_eval.to_csv(savename + '.csv', index=False)

    g = gt.Graph()
    # Set property maps for edge attributes
    weight = g.new_edge_property('float')

    # Add edges and nodes in bulk
    node_id = g.add_edge_list(pd_eval.values, hashed=True, eprops=[weight]) 
    g.vertex_properties['node_id'] = node_id
    g.edge_properties['weight'] = weight

    # Save graph
    g.save(savename + '.graphml')

if __name__ == '__main__':
    '''
    gl = pd.read_csv('graphs/graph_postCOVID.csv')
    gl.info(memory_usage = 'deep')

    print(err)
    '''
    print('Graph pre 2020')
    # list of rows where years are before and after 2020
    df_links = pd.read_csv('list_of_edges_pre2020.csv', converters={'list_of_edges': eval})
    pd_eval = computeWeights(df_links)
    createGraph(pd_eval, 'graph_preCOVID')
    del pd_eval


'''
    print('Graph post 2020')
    # list of rows where years are before and after 2020
    df_links = pd.read_csv('list_of_edges_post2020.csv', converters={'list_of_edges': eval})
    pd_eval = computeWeights(df_links)
    createGraph(pd_eval, 'graphs/graph_postCOVID')
    del pd_eval

    weights = g.ep["weight"]
print(list(weights))

name = g.vp["node_id"]
print(list(name))
print(len(list(name)))
'''