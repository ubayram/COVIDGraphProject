import numpy as np
import scipy.sparse as sp
import pandas as pd 
import networkx as nx

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_data(adj, prevent_disconnect=True, test_frac = 0.3):
	
	# print('creating adjacency matrix...')
	# adj = nx.adjacency_matrix(g)

	print('preprocessing...')

	# check number of graphs(or subgraphs) we want all the nodes of the graph should remain connected 
	g = nx.from_scipy_sparse_matrix(adj)
	orig_num_cc = nx.number_connected_components(g)

	adj_triu = sp.triu(adj) # upper triangular portion of adj matrix
	adj_tuple = sparse_to_tuple(adj_triu) # (coords, values, shape), edges only 1 way
	edges = adj_tuple[0] # all edges, listed only once (not 2 ways)

	# Store edges in list of ordered tuples (node1, node2) where node1 < node2
	edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
	all_edge_tuples = set(edge_tuples)
	train_edges = set(edge_tuples) # initialize train_edges to have all edges
	
	num_test = int(np.floor(len(train_edges) * test_frac)) # controls how large the test set for poitive edges should be
	test_edges = set()

	print('generating test sets...')

	# Iterate over shuffled edges, add to train/val sets
	np.random.shuffle(edge_tuples)
	count=0

	for edge in edge_tuples:
		
		# print edge
		node1 = edge[0]
		node2 = edge[1]

		# If removing edge would disconnect a connected component, backtrack and move on
		
		g.remove_edge(node1, node2)
		
		if prevent_disconnect == True:
			if nx.number_connected_components(g) > orig_num_cc:
				g.add_edge(node1, node2)
				continue
		# Fill test_edges first
		if len(test_edges) < num_test:
			test_edges.add(edge)
			train_edges.remove(edge)


		# Both edge lists full --> break loop
		elif len(test_edges) == num_test:
			break
		
		count+= 1
		print(count)

	if prevent_disconnect == True:
		assert nx.number_connected_components(g) == orig_num_cc

	print('creating false  edges...')

	edges_false = set()
	while len(edges_false) < num_test:
		idx_i = np.random.randint(0, adj.shape[0])
		idx_j = np.random.randint(0, adj.shape[0])
		if idx_i == idx_j:
			continue

		false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

		# Make sure false_edge not an actual edge, and not a repeat
		if false_edge in all_edge_tuples:
			continue
		if false_edge in edges_false:
			continue

		edges_false.add(false_edge)


	print('final checks for disjointness...')

	# assert: false_edges are actually false (not in all_edge_tuples)
	assert edges_false.isdisjoint(all_edge_tuples)

	# assert: test, val, train positive edges disjoint
	assert test_edges.isdisjoint(train_edges)

	print('creating adj_train...')

	# Re-build adj matrix using remaining graph
	adj_train = nx.adjacency_matrix(g)

	# Convert edge-lists to numpy arrays
	train_positive_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
	test_positive_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
	
	edges_false = np.array([list(edge_tuple) for edge_tuple in edges_false])
	num_test = int(np.floor(len(edges_false) * test_frac)) # controls how large the test set for poitive edges should be
	train_negative_edges = edges_false[:num_test]
	test_negative_edges = edges_false[num_test:]
	

	print('Done with train-test split!')
	print('')

	# NOTE: these edge lists only contain single direction of edge!
	return adj_train, train_positive_edges, test_positive_edges, train_negative_edges, test_negative_edges


def create_dataframes(train_positive_edges, train_negative_edges, test_positive_edges, test_negative_edges):
	positive_train_edges = pd.DataFrame(train_positive_edges, columns = ['node_1', 'node_2'])  
	positive_train_edges['label'] = 1

	negative_train_edges = pd.DataFrame(train_negative_edges, columns = ['node_1', 'node_2'])
	negative_train_edges['label'] = 0

	positive_test_edges = pd.DataFrame(test_positive_edges, columns = ['node_1', 'node_2'])
	positive_test_edges['label'] = 1

	negative_test_edges = pd.DataFrame(test_negative_edges, columns = ['node_1', 'node_2'])
	negative_test_edges['label'] = 0
	
	# concatenate the data
	Train_data = positive_train_edges.append(negative_train_edges)
	Test_data = positive_test_edges.append(negative_test_edges)

	return Train_data, Test_data

def prepare_graph(df_file):
	df = pd.read_csv(df_file)
	g = nx.from_pandas_edgelist(df, source='node1', target='node2', edge_attr='weight', create_using=nx.Graph())
	remove = [node for node,degree in dict(g.degree()).items() if degree <= 1]
	g.remove_nodes_from(remove)
	print('graph:  ', nx.info(g))
	return nx.adjacency_matrix(g)

def main():
	adj_mat = prepare_graph('./graph_postCOVID_final.csv')
	adj_train, train_positive_edges, test_positive_edges, train_negative_edges, test_negative_edges = preprocess_data(adj_mat, prevent_disconnect=True, test_frac = 0.3)
	Train_data, Test_data = create_dataframes(train_positive_edges, train_negative_edges, test_positive_edges, test_negative_edges)
	Train_data.to_csv('./Train_data.csv')
	Test_data.to_csv('./Test_data.csv')

if __name__ == main():
	main()
