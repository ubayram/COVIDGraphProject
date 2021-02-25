#import matplotlib.pyplot as plt
from math import isclose
#from sklearn import svm
import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import StellarGraph
from collections import Counter
import multiprocessing
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# Ulya Bayram
# ulyabayram@gmail.com
# Here Perform grid-search CV to find best p and q parameters optimizing the training data correctly predicting the validation set
# Define the other parameters globally
import itertools
from tensorflow import keras
import pickle

dimensions = 128
num_walks = 10
walk_length = 60
window_size = 12
workers = multiprocessing.cpu_count()

def node2vec_embedding(graph, name, p, q):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q, weighted=True)
    print(f"Number of weighted random walks for '{name}': {len(walks)}")

    model = Word2Vec(
        walks,
        size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        #iter=num_iter,
    )

    def get_embedding(u):
        return model.wv[u]

    return get_embedding

# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]

# inside, save the trained model to the corresponding folder - might be needed in the future
def trainTheClassifier(training_matrix, training_labels, validation_matrix, validation_labels):

    num_features = np.shape(training_matrix)[1]

    es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=25, verbose=0, mode='auto', restore_best_weights=True)# 

    # define the model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(100, activation='tanh', input_shape=(num_features,)))
    model.add(keras.layers.Dense(50, activation='tanh'))
    model.add(keras.layers.Dense(10, activation='tanh'))
    model.add(keras.layers.Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_crossentropy', 'accuracy'])

    # this fit part has a huge error that cannot be solved
    #scaler = StandardScaler()
    model.fit(StandardScaler().fit_transform(training_matrix), np.asarray(training_labels), epochs=200, batch_size=2500,
                  validation_data=(StandardScaler().fit_transform(validation_matrix), np.asarray(validation_labels)), callbacks=[es])

    return model

# 2. training classifier
def train_link_prediction_model(
    link_examples, link_labels, link_valexamples, validation_labels, binary_operator
):
    
    #link_features = link_examples_to_features(
    #    link_examples, get_embedding, binary_operator
    #)
    link_features = np.load('training_embeddings_' + binary_operator +'_split5.npz')['arr_0']

    #link_valfeatures = link_examples_to_features(
    #    link_valexamples, get_embedding, binary_operator
    #)
    link_valfeatures = np.load('val_embeddings_' + binary_operator +'_split5.npz')['arr_0']

    print('Training size')
    print(link_features.shape)
    print(len(link_labels))

    print('Val size')
    print(link_valfeatures.shape)
    print(len(validation_labels))
    #clf = link_prediction_classifier(link_features, link_labels, link_valfeatures, validation_labels)
    clf = trainTheClassifier(link_features, link_labels, link_valfeatures, validation_labels)
    #clf.fit(link_features, link_labels) # Error here
    return clf

#def link_prediction_classifier(link_features, link_labels, link_valfeatures, validation_labels):
#    lr_clf = trainTheClassifier(link_features, link_labels, link_valfeatures, validation_labels)
#    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

# 3. and 4. evaluate classifier
def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, binary_operator, filenametosave, readfilename
):
    #link_features_test = link_examples_to_features(
    #    link_examples_test, get_embedding, binary_operator
    #)
    link_features_test = np.load(readfilename)['arr_0']
    print('Test size')
    print(link_features_test.shape)
    print(len(link_labels_test))
    score_auc, score_acc = evaluate_roc_auc(clf, link_features_test, link_labels_test, filenametosave)

    return score_auc, score_acc

def computeAccuracy(list_pred_res, list_real_res):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    print('Before error check')
    print(len(list_pred_res))
    print(len(list_real_res))
    for i_ in range(len(list_pred_res)):
        # find out if the nodes are connected in the g_test
        is_connected = list_real_res[i_]
        prediction = list_pred_res[i_]
       # if is_connected and prediction == 1:
        if is_connected and prediction >= 0.5:
            tp += 1
        #elif is_connected and prediction == 0:
        elif is_connected and prediction < 0.5: 
            fn += 1
        #elif not is_connected and prediction == 0:
        elif not is_connected and prediction < 0.5:    
            tn += 1
        #elif not is_connected and prediction == 1:
        elif not is_connected and prediction >= 0.5:    
            fp += 1
        else:
            print('Error!')
        i_ += 1
    return (tp + tn)/float(tp+tn+fp+fn)

def evaluate_roc_auc(clf, link_features, link_labels, filename):
    predicted = clf.predict(StandardScaler().fit_transform(link_features))
    #predicted = clf.predict(link_features)
 
     # check which class corresponds to positive links
    positive_column = 1
    #positive_column = 1
    
    if len(filename) > 0: # save only when its not empty
        #Open and Write results to file.
        fo = open(filename+".txt","w")
        for i in range(len(predicted[:, positive_column])):
            fo.write(str(predicted[i, positive_column])+ '\t' + str(link_labels[i]) + '\n')

    #This was chancged 
    #score_acc = computeAccuracy(predicted, link_labels)
    score_acc = computeAccuracy(predicted[:, positive_column], link_labels)
    
    #score_acc = computeAccuracy(predicted4acc, link_labels)
    return roc_auc_score(link_labels, predicted[:, positive_column]), score_acc

def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0

binary_operators = [operator_hadamard, operator_l1, operator_avg]

def test_run_link_prediction(binary_operator):
    clf = train_link_prediction_model(
        examples_train, labels_train, examples_model_selection, labels_model_selection, binary_operator
    )
    score_auc, score_acc = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        binary_operator,
        '', # means don't save
        'val_embeddings_' + binary_operator +'_split5.npz'
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score_auc": score_auc,
        "score_acc": score_acc
    }#score_auc, score_acc

def findBestRandWalkParams():#ntxStgrph):

    results = [test_run_link_prediction(op) for op in list_names] #score_auc, score_acc = 
    currbest_result = max(results, key=lambda result: result["score_auc"])

    return currbest_result

'''
def run_link_prediction(binary_operator, filename2save):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score_auc, score_acc = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
        filename2save
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score_auc": score_auc,
        "score_acc": score_acc
    }
'''

def applyDeepLearning(save_dir):

    print('Load the data from files')

    #examples_train in the example .csv file
    #examples_train_df = pd.read_csv("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Train_Graphs/Graph_train_edges/graph_train_edges_sampled_1.csv")
    examples_train_df = pd.read_csv("../../graphs/graph_train_edges_sampled_5.csv")
    examples_train_df = examples_train_df.replace(np.nan, 'nan', regex=True)
    labels_train = list(examples_train_df['labels'])
    examples_train = [[i, j] for i, j in zip(list(examples_train_df['node1']), list(examples_train_df['node2']))]

    #----------------------------------------------------------------------------------------

    #examples_model_selection_df = pd.read_csv("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Validation_Graphs/Graph_validation_edges/graph_val_edges_sampled_1.csv")
    examples_model_selection_df = pd.read_csv("../../graphs/graph_val_edges_sampled_5.csv")

    examples_model_selection_df = examples_model_selection_df.replace(np.nan, 'nan', regex=True)
    labels_model_selection = list(examples_model_selection_df['labels'])
    examples_model_selection = [[i, j] for i, j in zip(list(examples_model_selection_df['node1']), list(examples_model_selection_df['node2']))]
    #----------------------------------------------------------------------------------------

    #examples_test_df = pd.read_csv("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Test_Graphs/Graph_test_edges/graph_test_edges_sampled_1.csv")
    examples_test_df = pd.read_csv("../../graphs/graph_test_edges_sampled_5.csv")

    examples_test_df = examples_test_df.replace(np.nan, 'nan', regex=True)
    labels_test = list(examples_test_df['labels'])
    examples_test = [[i, j] for i, j in zip(list(examples_test_df['node1']), list(examples_test_df['node2']))]
    #----------------------------------------------------------------------------------------

    #graph_to_embed = ("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Train_Graphs/graph_sampled_1.gml.gz")
    #graph_to_embed = ("graphs/graph_sampled_5.gml.gz")
    #g = nx.read_gml(graph_to_embed)

    #Assign networkx data utilizing Stellargraph 
    #ntxStgrph = StellarGraph.from_networkx(g)

    #curr_embedding = node2vec_embedding(ntxStgrph, "Train Graph", 0.25, 0.25)

    list_names = ['op_hadamard', 'op_l1', 'op_avg'] #[operator_hadamard, operator_l1, operator_avg]
    '''
    for i_op in range(len(binary_operators)):
        binary_operator = binary_operators[i_op]
        curr_name = list_names[i_op]

        # convert training set
        converted_training_matrix = link_examples_to_features(examples_train, curr_embedding, binary_operator)

        # convert validation set
        converted_val_matrix = link_examples_to_features(examples_model_selection, curr_embedding, binary_operator)

        # convert test set
        converted_test_matrix = link_examples_to_features(examples_test, curr_embedding, binary_operator)

        # convert them to numpy matrices and save them
        converted_training_matrix = np.array(converted_training_matrix)
        converted_val_matrix = np.array(converted_val_matrix)
        converted_test_matrix = np.array(converted_test_matrix)

        np.savez_compressed('training_embeddings_' + list_names[i_op] +'_split5.npz', converted_training_matrix)
        np.savez_compressed('val_embeddings_' + list_names[i_op] +'_split5.npz', converted_val_matrix)
        np.savez_compressed('test_embeddings_' + list_names[i_op] +'_split5.npz', converted_test_matrix)
    
    '''

    print('\nGet the best fitting random walk parameters based on pre-selected others')
    best_result = findBestRandWalkParams()#ntxStgrph)

    print('Best result ' + str(best_result))
    #print('Best q parameter ' + str(q))
    #print('Best binary operator ' + str(best_result['binary_operator'].__name__))

    print('\nEmbedding is ready, collect performance evaluation results')
    test_score_auc, test_score_acc = evaluate_link_prediction_model(
        best_result["classifier"],
        examples_test,
        labels_test,
        #embedding_train,
        best_result["binary_operator"],
        save_dir + 'predicted_test_result_mp_nogrid_split5',
        'test_embeddings_' + best_result['binary_operator'] +'_split5.npz'
    )
    print(f"ROC AUC score on test set using '{best_result['binary_operator']}': {test_score_auc}")
    print(f"Accuracy score on test set using '{best_result['binary_operator']}': {test_score_acc}")


def applyNaiveBayes(save_dir):

    print('Load the data from files')

    #examples_train in the example .csv file
    #examples_train_df = pd.read_csv("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Train_Graphs/Graph_train_edges/graph_train_edges_sampled_1.csv")
    examples_train_df = pd.read_csv("../../graphs/graph_train_edges_sampled_5.csv")
    examples_train_df = examples_train_df.replace(np.nan, 'nan', regex=True)
    labels_train = list(examples_train_df['labels'])
    examples_train = [[i, j] for i, j in zip(list(examples_train_df['node1']), list(examples_train_df['node2']))]

    #----------------------------------------------------------------------------------------

    #examples_model_selection_df = pd.read_csv("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Validation_Graphs/Graph_validation_edges/graph_val_edges_sampled_1.csv")
    examples_model_selection_df = pd.read_csv("../../graphs/graph_val_edges_sampled_5.csv")

    examples_model_selection_df = examples_model_selection_df.replace(np.nan, 'nan', regex=True)
    labels_model_selection = list(examples_model_selection_df['labels'])
    examples_model_selection = [[i, j] for i, j in zip(list(examples_model_selection_df['node1']), list(examples_model_selection_df['node2']))]
    #----------------------------------------------------------------------------------------

    #examples_test_df = pd.read_csv("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Test_Graphs/Graph_test_edges/graph_test_edges_sampled_1.csv")
    examples_test_df = pd.read_csv("../../graphs/graph_test_edges_sampled_5.csv")

    examples_test_df = examples_test_df.replace(np.nan, 'nan', regex=True)
    labels_test = list(examples_test_df['labels'])
    examples_test = [[i, j] for i, j in zip(list(examples_test_df['node1']), list(examples_test_df['node2']))]
    #----------------------------------------------------------------------------------------

    #graph_to_embed = ("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Train_Graphs/graph_sampled_1.gml.gz")
    graph_to_embed = ("../../graphs/graph_sampled_5.gml.gz")
    g = nx.read_gml(graph_to_embed)

    #Assign networkx data utilizing Stellargraph 
    ntxStgrph = StellarGraph.from_networkx(g)

    print('\nGet the best fitting random walk parameters based on pre-selected others')
    p, q, best_result, embedding_train = findBestRandWalkParams(ntxStgrph)

    print('Best p parameter ' + str(p))
    print('Best q parameter ' + str(q))
    print('Best binary operator ' + str(best_result['binary_operator'].__name__))

    print('\nEmbedding is ready, collect performance evaluation results')
    test_score_auc, test_score_acc = evaluate_link_prediction_model(
        best_result["classifier"],
        examples_test,
        labels_test,
        embedding_train,
        best_result["binary_operator"],
        save_dir + 'predicted_test_result_bayes_nogrid_split5' # Aqil, change the directory here so you won't forget now on
    )
    print(f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score_auc}")
    print(f"Accuracy score on test set using '{best_result['binary_operator'].__name__}': {test_score_acc}")

def applyLogisticRegression(save_dir):

    print('Load the data from files')

    #examples_train in the example .csv file
    #examples_train_df = pd.read_csv("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Train_Graphs/Graph_train_edges/graph_train_edges_sampled_1.csv")
    examples_train_df = pd.read_csv("../../graphs/graph_train_edges_sampled_5.csv")
    examples_train_df = examples_train_df.replace(np.nan, 'nan', regex=True)
    labels_train = list(examples_train_df['labels'])
    examples_train = [[i, j] for i, j in zip(list(examples_train_df['node1']), list(examples_train_df['node2']))]

    #----------------------------------------------------------------------------------------

    #examples_model_selection_df = pd.read_csv("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Validation_Graphs/Graph_validation_edges/graph_val_edges_sampled_1.csv")
    examples_model_selection_df = pd.read_csv("../../graphs/graph_val_edges_sampled_5.csv")

    examples_model_selection_df = examples_model_selection_df.replace(np.nan, 'nan', regex=True)
    labels_model_selection = list(examples_model_selection_df['labels'])
    examples_model_selection = [[i, j] for i, j in zip(list(examples_model_selection_df['node1']), list(examples_model_selection_df['node2']))]
    #----------------------------------------------------------------------------------------

    #examples_test_df = pd.read_csv("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Test_Graphs/Graph_test_edges/graph_test_edges_sampled_1.csv")
    examples_test_df = pd.read_csv("../../graphs/graph_test_edges_sampled_5.csv")

    examples_test_df = examples_test_df.replace(np.nan, 'nan', regex=True)
    labels_test = list(examples_test_df['labels'])
    examples_test = [[i, j] for i, j in zip(list(examples_test_df['node1']), list(examples_test_df['node2']))]
    #----------------------------------------------------------------------------------------

    #graph_to_embed = ("/home/ubuntu/ssl/COVID_Data/Updated Paper Graph_node2vec/Train_Graphs/graph_sampled_1.gml.gz")
    graph_to_embed = ("../../graphs/graph_sampled_5.gml.gz")
    g = nx.read_gml(graph_to_embed)

    #Assign networkx data utilizing Stellargraph 
    ntxStgrph = StellarGraph.from_networkx(g)

    print('\nGet the best fitting random walk parameters based on pre-selected others')
    p, q, best_result, embedding_train = findBestRandWalkParams(ntxStgrph)

    print('Best p parameter ' + str(p))
    print('Best q parameter ' + str(q))
    print('Best binary operator ' + str(best_result['binary_operator'].__name__))

    print('\nEmbedding is ready, collect performance evaluation results')
    test_score_auc, test_score_acc = evaluate_link_prediction_model(
        best_result["classifier"],
        examples_test,
        labels_test,
        embedding_train,
        best_result["binary_operator"],
        save_dir + 'predicted_test_result_logit_nogrid_split5_'
    )
    print(f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score_auc}")
    print(f"Accuracy score on test set using '{best_result['binary_operator'].__name__}': {test_score_acc}")