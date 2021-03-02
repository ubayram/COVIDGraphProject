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
# This code applies simple link prediction methods on the given graph
import networkx as nx
import pandas as pd
import numpy as np
#from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from tensorflow import keras

def nodes_connected(G, u, v):
    return u in G.neighbors(v)

def collectAdarScores(G, train_edges_name):
    
    # find which edges are unconnected in the training
    df_train = pd.read_csv(train_edges_name)
    df_train = df_train.replace(np.nan, 'nan', regex=True)
    #print(G.neighbors('nan'))
    #print(err)
    #list_unconnected = df_train.index[df_train['training_labels'] == 0].tolist() #df_train.where(df_train['training_labels']==0))

    list_real_labels = []
    list_pred_scores = []
    for i_row in range(len(df_train.node1)): # for each training set data
        node1 = df_train.node1[i_row]
        node2 = df_train.node2[i_row]

        # Find all nbrs of node1 and node2 in training graph that overlap
        list_nbrs = sorted(nx.common_neighbors(G, node1, node2))
            
        total_sum = 0
        # if list_nbrs isn't empty, find the weights of all the edges connected to the nbrs
        for i in range(len(list_nbrs)):
            curr_weight = G.degree(list_nbrs[i], weight='weight')
            total_sum += -1/np.log(curr_weight)

        #
        list_real_labels.append(df_train.labels[i_row])
        list_pred_scores.append(total_sum)

    return list_pred_scores, list_real_labels

def computeAccuracy(list_pred_res, list_real_res):

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i_ in range(len(list_pred_res)):

        # find out if the nodes are connected in the g_test
        is_connected = list_real_res[i_]
        prediction = list_pred_res[i_, 1]

        if is_connected and prediction >= 0.5:
            tp += 1
        elif is_connected and prediction < 0.5:
            fn += 1
        elif not is_connected and prediction < 0.5:
            tn += 1
        elif not is_connected and prediction >= 0.5:
            fp += 1
        else:
            print('Error!')

        i_ += 1
    
    return tp, tn, fp, fn

def applyNaiveBayes(save_dir, method_):

    fo = open(save_dir + 'accuracy_test_results_baseline_naive_bayes.txt', 'w')
    fo.write('set\ttp\ttn\tfp\tfn\tacc\n')

    foval = open(save_dir + 'accuracy_val_results_baseline_naive_bayes.txt', 'w')
    fo.write('set\ttp\ttn\tfp\tfn\tacc\n')
    # read the A_train graph
    for ix in range(1, 6):
        
        # ---------------- training ----------------------
        train_graph_name = '../../graphs/graph_sampled_' + str(ix) + '.gml.gz'
        G = nx.read_gml(train_graph_name)

        train_edges_name = '../../graphs/graph_train_edges_sampled_' + str(ix) + '.csv'

        list_pred_scores_train, list_real_labels_train = collectAdarScores(G, train_edges_name)

        #Create a Gaussian Classifier
        model = GaussianNB()

        # Train the model using the training sets
        model.fit(np.array(list_pred_scores_train).reshape(-1, 1),list_real_labels_train) # which score corresponds to which label

        # --------------- validation ---------------------------
        val_edges_name = '../../graphs/graph_val_edges_sampled_' + str(ix) + '.csv'

        # compute the performance over validation set
        list_pred_scores_val, list_real_labels_val = collectAdarScores(G, val_edges_name)
        list_pred_labels_val = model.predict_proba(np.array(list_pred_scores_val).reshape(-1, 1))

        tp_v, tn_v, fp_v, fn_v = computeAccuracy(list_pred_labels_val, list_real_labels_val)
        acc_v = (tp_v+tn_v)/float(tp_v+tn_v+fp_v+fn_v)
        foval.write('val\t' + str(tp_v) + '\t' + str(tn_v) + '\t' + str(fp_v) + '\t' + str(fn_v) + '\t' + str(acc_v) + '\n')
        # --------------- test ------------------------------------
        #test_graph_name = 'graphs/graph_test_' + str(ix) + '.gml.gz'
        test_edges_name = '../../graphs/graph_test_edges_sampled_' + str(ix) + '.csv'

        if 'adar' in method_:
            list_pred_scores_test, list_real_labels_test = collectAdarScores(G, test_edges_name)
        else:
            list_pred_scores_test, list_real_labels_test = collectJaccardScores(G, test_edges_name)
        list_pred_labels_test = model.predict_proba(np.array(list_pred_scores_test).reshape(-1, 1))

        fo__ = open(save_dir + 'prediction_results_test_naivebayes_' + str(ix) + '.txt', 'w')
        for icurrpred in range(len(list_pred_labels_test)):
            fo__.write(str(list_pred_labels_test[icurrpred, 1]) + '\t' + str(list_real_labels_test[icurrpred]) + '\n')
        fo__.close()
        # compute the performance over test set
        tp_t, tn_t, fp_t, fn_t = computeAccuracy(list_pred_labels_test, list_real_labels_test)
        acc_t = (tp_t+tn_t)/float(tp_t+tn_t+fp_t+fn_t)
        fo.write('test\t' + str(tp_t) + '\t' + str(tn_t) + '\t' + str(fp_t) + '\t' + str(fn_t) + '\t' + str(acc_t) + '\n')
        del G

    fo.close()
    foval.close()

def applyLogisticRegression(save_dir, method_):

    fo = open(save_dir + 'accuracy_test_results_baseline_logit.txt', 'w')
    fo.write('set\ttp\ttn\tfp\tfn\tacc\n')

    foval = open(save_dir + 'accuracy_val_results_baseline_logit.txt', 'w')
    fo.write('set\ttp\ttn\tfp\tfn\tacc\n')
    # read the A_train graph
    for ix in range(1, 6):
        
        # ---------------- training ----------------------
        train_graph_name = '../../graphs/graph_sampled_' + str(ix) + '.gml.gz'
        G = nx.read_gml(train_graph_name)

        train_edges_name = '../../graphs/graph_train_edges_sampled_' + str(ix) + '.csv'

        list_pred_scores_train, list_real_labels_train = collectAdarScores(G, train_edges_name)

        #Create LR classifier
        model = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=2000)

        # Train the model using the training sets
        model.fit(np.array(list_pred_scores_train).reshape(-1, 1),list_real_labels_train) # which score corresponds to which label

        # --------------- validation ---------------------------
        val_edges_name = '../../graphs/graph_val_edges_sampled_' + str(ix) + '.csv'

        # compute the performance over validation set
        list_pred_scores_val, list_real_labels_val = collectAdarScores(G, val_edges_name)
        list_pred_labels_val = model.predict_proba(np.array(list_pred_scores_val).reshape(-1, 1))

        tp_v, tn_v, fp_v, fn_v = computeAccuracy(list_pred_labels_val, list_real_labels_val)
        acc_v = (tp_v+tn_v)/float(tp_v+tn_v+fp_v+fn_v)
        foval.write('val\t' + str(tp_v) + '\t' + str(tn_v) + '\t' + str(fp_v) + '\t' + str(fn_v) + '\t' + str(acc_v) + '\n')
        # --------------- test ------------------------------------
        #test_graph_name = 'graphs/graph_test_' + str(ix) + '.gml.gz'
        test_edges_name = '../../graphs/graph_test_edges_sampled_' + str(ix) + '.csv'

        if 'adar' in method_:
            list_pred_scores_test, list_real_labels_test = collectAdarScores(G, test_edges_name)
        else:
            list_pred_scores_test, list_real_labels_test = collectJaccardScores(G, test_edges_name)
        list_pred_labels_test = model.predict_proba(np.array(list_pred_scores_test).reshape(-1, 1))

        fo__ = open(save_dir + 'prediction_results_test_logit_split' + str(ix) + '.txt', 'w')
        for icurrpred in range(len(list_pred_labels_test)):
            fo__.write(str(list_pred_labels_test[icurrpred, 1]) + '\t' + str(list_real_labels_test[icurrpred]) + '\n')
        fo__.close()
        # compute the performance over test set
        tp_t, tn_t, fp_t, fn_t = computeAccuracy(list_pred_labels_test, list_real_labels_test)
        acc_t = (tp_t+tn_t)/float(tp_t+tn_t+fp_t+fn_t)
        fo.write('test\t' + str(tp_t) + '\t' + str(tn_t) + '\t' + str(fp_t) + '\t' + str(fn_t) + '\t' + str(acc_t) + '\n')
        del G

    fo.close()
    foval.close()

# inside, save the trained model to the corresponding folder - might be needed in the future
def trainTheClassifier(training_matrix, training_labels, validation_matrix, validation_labels):

    num_features = np.shape(training_matrix)[1]

    # define the early stopping criteria
    es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=25, verbose=0, mode='auto', restore_best_weights=True)# 

    # define the model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(100, activation='tanh', input_shape=(num_features,)))
    #model.add(keras.layers.Dropout(0.25))
    #model.add(keras.layers.Dense(80, activation='relu', kernel_initializer='he_normal'))
    #model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(50, activation='tanh'))
    #model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(10, activation='tanh'))
    #model.add(keras.layers.Dropout(0.25))
    #model.add(keras.layers.Dense(5, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(2, activation='sigmoid'))

    # with sgd optimizer, the result was 0.74, i just replaced it with adam and got 0.88 - the highest performance so far
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_crossentropy', 'accuracy'])

    model.fit(training_matrix, np.asarray(training_labels), epochs=200, batch_size=2500,
                  validation_data=(validation_matrix, np.asarray(validation_labels)), callbacks=[es])

    return model

def classifyFeatures(input_matrix, trained_model):

    probas_ = trained_model.predict(input_matrix)

    return probas_

def applyDeepLearning(save_dir, method_):

    fo = open(save_dir + 'accuracy_test_results_baseline_mp.txt', 'w')
    fo.write('set\ttp\ttn\tfp\tfn\tacc\n')

    #foval = open('accuracy_val_results_baseline_mp.txt', 'w')
    #fo.write('set\ttp\ttn\tfp\tfn\tacc\n')
    # read the A_train graph
    for ix in range(1, 6):#range(1, 6)
        
        # ---------------- training ----------------------
        train_graph_name = '../../graphs/graph_sampled_' + str(ix) + '.gml.gz'
        G = nx.read_gml(train_graph_name)

        train_edges_name = '../../graphs/graph_train_edges_sampled_' + str(ix) + '.csv'

        list_pred_scores_train, list_real_labels_train = collectAdarScores(G, train_edges_name)

        # --------------- validation ---------------------------
        val_edges_name = '../../graphs/graph_val_edges_sampled_' + str(ix) + '.csv'

        # compute the performance over validation set
        list_pred_scores_val, list_real_labels_val = collectAdarScores(G, val_edges_name)
        
        # train and fit the deep learning model np.array(list_pred_scores).reshape(-1, 1)
        model = trainTheClassifier(np.array(list_pred_scores_train).reshape(-1, 1), list_real_labels_train, np.array(list_pred_scores_val).reshape(-1, 1), list_real_labels_val)

        # --------------- test ------------------------------------
        #test_graph_name = 'graphs/graph_test_' + str(ix) + '.gml.gz'
        test_edges_name = '../../graphs/graph_test_edges_sampled_' + str(ix) + '.csv'

        if 'adar' in method_:
            list_pred_scores_test, list_real_labels_test = collectAdarScores(G, test_edges_name)
        else:
            list_pred_scores_test, list_real_labels_test = collectJaccardScores(G, test_edges_name)
        list_pred_labels_test = classifyFeatures(np.array(list_pred_scores_test).reshape(-1, 1), model)

        fo__ = open(save_dir + 'prediction_results_test_mp_' + str(ix) + '.txt', 'w')
        for icurrpred in range(len(list_pred_labels_test)):
            fo__.write(str(list_pred_labels_test[icurrpred, 1]) + '\t' + str(list_real_labels_test[icurrpred]) + '\n')
        fo__.close()
        # compute the performance over test set
        tp_t, tn_t, fp_t, fn_t = computeAccuracy(list_pred_labels_test[:, 1], list_real_labels_test)
        acc_t = (tp_t+tn_t)/float(tp_t+tn_t+fp_t+fn_t)
        fo.write('test\t' + str(tp_t) + '\t' + str(tn_t) + '\t' + str(fp_t) + '\t' + str(fn_t) + '\t' + str(acc_t) + '\n')
        del G

    fo.close()
    #foval.close()

def collectJaccardScores(G, train_edges_name):
    
    # find which edges are unconnected in the training
    df_train = pd.read_csv(train_edges_name)
    df_train = df_train.replace(np.nan, 'nan', regex=True)
    #print(G.neighbors('nan'))
    #print(err)
    #list_unconnected = df_train.index[df_train['training_labels'] == 0].tolist() #df_train.where(df_train['training_labels']==0))

    list_real_labels = []
    list_pred_scores = []
    for i_row in range(len(df_train.node1)): # for each training set data
        node1 = df_train.node1[i_row]
        node2 = df_train.node2[i_row]

        # Find all nbrs of node1 and node2 in training graph that overlap
        list_commonnbrs = sorted(nx.common_neighbors(G, node1, node2))
        
        sum_common = 0
        # if list_nbrs isn't empty, find the weights of all the edges connected to the nbrs
        for i in range(len(list_commonnbrs)):
            curr_weight = G.degree(list_commonnbrs[i], weight='weight')
            sum_common += curr_weight
        
        list_allnbrs1 = list(G.neighbors(node1))#[n for n in G.neighbors(node1)]
        list_allnbrs2 = list(G.neighbors(node2))#[n for n in G.neighbors(node2)] #[n for n in G.neighbors(0)]
        list_mergednbrs = list(set(list_allnbrs1 + list_allnbrs2))
        #print('Num node1 ' + str(len(list_allnbrs1)))
        #print('Num node2 ' + str(len(list_allnbrs2)))
        #print('Num total ' + str(len(list_mergednbrs)))

        sum_union = 0
        # if list_nbrs isn't empty, find the weights of all the edges connected to the nbrs
        for i in range(len(list_mergednbrs)):
            curr_weight = G.degree(list_mergednbrs[i], weight='weight')
            sum_union += curr_weight
        
        if sum_union > 0:
            total_res = sum_common/float(sum_union)
        else:
            total_res = 0
        #
        list_real_labels.append(df_train.labels[i_row])
        list_pred_scores.append(total_res)

    return list_pred_scores, list_real_labels
