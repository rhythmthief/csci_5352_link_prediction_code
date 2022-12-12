from sklearn import metrics
import pandas as pd
import networkx as nx
import numpy as np
import statistics as st
from multiprocessing import Pool
import matplotlib.pyplot as plt
from numpy import trapz
import sys
import random
from datetime import datetime
import os
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # networkx has a future warning I don't need to see

random.seed(1337)  # for consistency


def jaccard_MT(G, edges_and_non_edges):
    # compute jaccard coefficient for every edge and non-edge in the graph
    # I am doing the actual edges here as well because I figured
    # it could be useful for the stacking model learning

    scores = []  # stores prediction scores
    # upper bound on the noise so that it is much smaller than 1/(n-2)

    args = list(zip([G for _ in range(len(edges_and_non_edges))], [
                e for e in edges_and_non_edges]))

    scores = Pool().map(jaccard_MT_Helper, args)

    return scores


def jaccard_MT_Helper(args):
    G = args[0]
    e = args[1]
    noise_bound = (1 / (G.number_of_nodes() - 2)) / 10

    e0_neighbors = list(G.neighbors(e[0]))
    e1_neighbors = list(G.neighbors(e[1]))

    # compute number of elts in the intersection of two neighborhoods
    num = len([n for n in e0_neighbors if n in e1_neighbors])

    # add lists together, reduce to a set with no duplicates
    den = len(set(e0_neighbors + e1_neighbors))

    if den == 0:
        # avoid division by zero
        # if union is empty, intersection is likewise empty anyway
        # so now we gracefully divide 0 by 1
        den = 1

    return float(num / den + np.random.uniform(0, noise_bound, 1))


def shortest_path_attr_MT(G, edges_and_non_edges, attr):
    # compute shortest path topological predictor scores
    # also computes weighted version from extra credit for HW3
    # and just in case it helps, returns the weight scores as well

    # find every non-edge in the graph

    # arguments for multiprocessing
    args = list(zip([G for _ in range(len(edges_and_non_edges))],
                [e for e in edges_and_non_edges], [attr for _ in range(len(edges_and_non_edges))]))

    # compute scores on multiple threads
    scores = np.array(Pool().map(shortest_path_attr_MT_Helper, args))

    scores_sp = scores[:, 0]
    scores_attr = scores[:, 1]
    scores_sp_attr = scores[:, 2]

    return scores_sp, scores_attr, scores_sp_attr


def shortest_path_attr_MT_Helper(args):
    G = args[0]
    e = args[1]
    attr = args[2]
    # smaller than 1 / (longest possible path)
    noise_bound = (1 / (G.number_of_nodes() - 1)) / 10

    try:
        sp = nx.shortest_path_length(G, source=e[0], target=e[1])
    except nx.NetworkXNoPath:
        # shortest_path_length does not assign inf by default and throws an error if no path exists
        sp = float('inf')

    if attr != '':
        attr_score = shortest_path_attr_compute_helper(G, e, attr)
    else:
        attr_score = 0

    return [float(1/sp + np.random.uniform(0, noise_bound, 1)), attr_score, float(1/sp + attr_score + np.random.uniform(0, noise_bound, 1))]


def shortest_path_attr_compute_helper(G, edge, attr):
    # returns additional scoring based on whether attributes from either side of edges sampled uniformly at random were the same

    # choose n random edges that aren't missing
    # if the pair of attributes on either side is the same as
    # the pair of attributes on nodes connected by the considered edge
    # add 1 to score

    n_r = 100  # number of edges to scan
    score = 0

    edgeList = list(G.edges())

    a0 = G.nodes[edge[0]][attr]
    a1 = G.nodes[edge[1]][attr]
    edgenum = G.number_of_edges()

    # sample n_r observed edges uniformly at random

    if edgenum <= n_r:
        n_r = edgenum  # reduce n_r choices in case too many edges were removed

    edge_ix = np.random.choice(edgenum, n_r, replace=False)

    for ix in edge_ix:
        e = edgeList[ix]
        a0e = G.nodes[e[0]][attr]
        a1e = G.nodes[e[1]][attr]

        if (a0 == a0e and a1 == a1e) or (a0 == a1e and a1 == a0e):
            score = score + 1

    return score / n_r


def Katz_centrality_MT(G, edges_and_non_edges):
    # katz centrality scores for nodes on either side of the edge
    # by themselves, probably aren't good predictors
    # but could serve as good topological features for the stacking model
    # to learn from

    katz_generator = nx.katz_centrality_numpy(G)

    # arguments for multiprocessing
    args = list(zip([katz_generator for _ in range(len(edges_and_non_edges))],
                [e for e in edges_and_non_edges]))

    katz = np.array(Pool().map(Katz_centrality_MT_Helper, args))

    return katz[:, 0], katz[:, 1]

def Katz_centrality_MT_Helper(args):
    katz_generator = args[0]
    e = args[1]
    return katz_generator[e[0]], katz_generator[e[1]]

def degree_product_MT(G, edges_and_non_edges):
    # compute degree product topological predictor scores
    # also returns degrees themselves

    scores = []  # stores prediction scores

    # arguments for multiprocessing
    args = list(zip([G for _ in range(len(edges_and_non_edges))],
                [e for e in edges_and_non_edges]))

    # compute scores on multiple threads
    scores = np.array(Pool().map(degree_product_MT_Helper, args))

    return scores


def degree_product_MT_Helper(args):
    G = args[0]
    e = args[1]

    return G.degree[e[0]], G.degree[e[1]], float(G.degree[e[0]] * G.degree[e[1]] +
                                                 np.random.uniform(0, 0.0001, 1))  # noise is epsilon << 1


def adamic_adar_MT(G, edges_and_non_edges):
    # arguments for multiprocessing
    args = list(zip([G for _ in range(len(edges_and_non_edges))],
                [e for e in edges_and_non_edges]))

    # compute scores on multiple threads
    scores = Pool().map(adamic_adar_MT_Helper, args)

    return scores


def adamic_adar_MT_Helper(args):
    G = args[0]
    e = args[1]

    return sum(1 / np.log(G.degree(w)) for w in nx.common_neighbors(G, e[0], e[1])) + float(np.random.uniform(0, 0.0001, 1))


def common_neighbor_centrality_MT(G, edges_and_non_edges):
    # CCPA score
    # MT reimplementation of nx's code
    # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_prediction.html

    # shortest path length object to be passed in
    spl = dict(nx.shortest_path_length(G))

    args = list(zip([G for _ in range(len(edges_and_non_edges))],
                [e for e in edges_and_non_edges], [spl for _ in range(len(edges_and_non_edges))]))

    # compute scores on multiple threads
    scores = Pool().map(common_neighbor_centrality_MT_Helper, args)

    return scores


def common_neighbor_centrality_MT_Helper(args):
    G = args[0]
    e = args[1]
    spl = args[2]
    inf = float("inf")
    alpha = 0.8  # the author claimed that this is the optimal alpha value

    # this code is basically from NX's implementation of this predictor
    # with minimal adjustments like noise
    if e[0] == e[1]:
        raise nx.NetworkXAlgorithmError("Self links are not supported")
    path_len = spl[e[0]].get(e[1], inf)

    return alpha * sum(1 for _ in nx.common_neighbors(G, e[0], e[1])) + (1 - alpha) * (G.number_of_nodes() / path_len) + float(np.random.uniform(0, 0.0001, 1))


def attribute_match_MT(G, edges_and_non_edges, attr):
    # records whether nodes on either side of the potential edge have the same focal attribute
    # return 1
    # 0

    scores = []

    if attr != '':
        # only do this if there is an attribute being considered
        args = list(zip([(G.nodes[e[0]][attr], G.nodes[e[1]][attr])
                    for e in edges_and_non_edges]))

        # compute scores on multiple threads
        scores = Pool().map(attribute_match_MT_Helper, args)

    return scores


def attribute_match_MT_Helper(args):
    attr_tuple = args[0]
    score = 0
    if attr_tuple[0] == attr_tuple[1]:
        score = 1
    return score


def resource_allocation_index_MT(G, edges_and_non_edges):
    # computes resource allocation index scores
    # MT version of the networkx implementation
    # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_prediction.html#resource_allocation_index

    args = list(zip([G for _ in range(len(edges_and_non_edges))],
                [e for e in edges_and_non_edges]))

    # compute scores on multiple threads
    scores = Pool().map(resource_allocation_index_MT_Helper, args)

    return scores


def resource_allocation_index_MT_Helper(args):
    G = args[0]
    e = args[1]
    return sum(1 / G.degree(w) for w in nx.common_neighbors(G, e[0], e[1])) + float(np.random.uniform(0, 0.00001, 1))


def betweenness_centrality_MT(G, edges_and_non_edges):
    bc = nx.betweenness_centrality(G, normalized=True)

    args = list(zip([G for _ in range(len(edges_and_non_edges))],
                [e for e in edges_and_non_edges], [bc for _ in range(len(edges_and_non_edges))]))

    # compute scores on multiple threads
    scores = np.array(Pool().map(betweenness_centrality_MT_Helper, args))

    return scores


def betweenness_centrality_MT_Helper(args):
    # extract betweenness centrality for nodes on either side of the edge
    G = args[0]
    e = args[1]
    bc = args[2]

    return [bc[e[0]],  bc[e[1]]]

# dataframe for AUC computations
# list of features
# total number of true erased edges

def AUC_MT(df, features, T):
    Y = len(df)
    F = Y - T  # true negatives
    results = {}

    for feat in features:
        # sort by score in descending order
        df.sort_values(feat, ascending=False, inplace=True)

        t = df[['t']].to_numpy()  # read sorted t column back out

        # list of arguments for parallelized AUC computations
        args = list(zip([i for i in range(Y)], [t for _ in range(Y)], [
                    T for _ in range(Y)], [F for _ in range(Y)]))

        # compute vals for AUC on multiple threads
        # results come back ordered so no need to sort
        res = np.array(Pool().map(AUC_MT_HELPER, args))

        TPR = res[:, 1]
        FPR = res[:, 2]
        AUC = trapz(TPR, FPR)  # integrate using trapezoidal rule
        results[feat] = [AUC, FPR, TPR]

    return results


def AUC_MT_HELPER(args):
    # helper for parallelized computations

    i = args[0]
    t = args[1]
    T = args[2]
    F = args[3]

    tsum = np.sum(t[:i])
    TPR = (1/T) * tsum
    # adding 1 because indexing is natural in the formula
    FPR = (1/F) * (i + 1 - tsum)

    return [i, TPR, FPR]

def visualizeDrugNetwork(G, edges_removed, edges_recovered):
    # prepare visualization

    # https://networkx.org/documentation/stable/auto_examples/
    # drawing/plot_labels_and_colors.html
    pos = nx.spring_layout(G, seed=3113794652) # seed is from 
    colors = ["tab:red", "tab:blue", "tab:green", "tab:cyan", "tab:gray"]

    # nodelists for states and families
    states = list(G.nodes())[0:32]
    families = list(G.nodes())[32:42]

    nx.draw_networkx_nodes(G, pos, nodelist=states, node_color=colors[3], node_size=800)
    nx.draw_networkx_nodes(G, pos, nodelist=families, node_color=colors[4], node_size=800)

    # edges
    #nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    predicted = []
    not_predicted = []

    for e in edges_removed:
        if e in edges_recovered:
            predicted.append(e)
        else:
            not_predicted.append(e)
    
    nx.draw_networkx_edges(G, pos, edgelist=predicted, edge_color=colors[1], width=1.0, alpha=0.5, label='predicted')
    nx.draw_networkx_edges(G, pos, edgelist=not_predicted, edge_color=colors[0], width=1.0, alpha=0.5, label='not predicted')

    # some math labels
    labels = {}

    for i in range(len(G.nodes())):
        labels[i] = i+1

    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color="whitesmoke")

    plt.tight_layout()
    #plt.title("$\mathcal{L} =$" + str(round(L,2)), fontsize="22")
    plt.legend(loc="best", prop={'size': 25})

    plt.axis("off")
    plt.show()

def plotROC(features, AUC_vals):
    # plot ROC curve for select features

    plt.xlabel('False positive rate (FPR)', fontsize=30)
    plt.ylabel('True positive rate (TPR)', fontsize=30)
    plt.title('ROC Curves for various predictors', fontsize=30)

    plt.plot([0, 1], [0, 1], c='grey', linestyle='dashed')

    # randomized colors
    # for feat in features:
    #     plt.plot(AUC_vals[feat][1], AUC_vals[feat][2],
    #              c=np.random.rand(3,), label=feat)

    # stacking is red, others are grey
    for feat in features:

        if feat == 'stacking':
            color = 'red'
        else:
            color = 'grey'

        plt.plot(AUC_vals[feat][1], AUC_vals[feat][2],
                 c=color, label=feat)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="best", prop={'size': 25})
    plt.show()

def plotBoxPlot(features, auc_runs):
    data = []

    for i in range(len(features)):
        data.append(auc_runs[:,i])

    #fig = plt.figure(figsize =(10, 7))

    # Creating axes instance
    #ax = fig.add_axes([0, 0, 1, 1])
    
    fig, ax = plt.subplots()
    
    plt.title('AUC for various link predictors', fontsize=30)

    #plt.plot([0, 1], [0, 1], c='grey', linestyle='dashed')
    plt.axhline(y=1.0, color='grey', linestyle='dashed')

    # some relabeling as the old labels don't look good on the plot
    features[features.index('shortest_path')] = 'sh_path'
    features[features.index('degree_product')] = 'deg_prod'
    features[features.index('adamic_adar')] = 'ad_ad'

    # Creating plot
    ax.boxplot(data, labels=features)
    plt.xticks(range(1, len(features)+1), features, fontsize=20)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    # plt.legend(loc="best", prop={'size': 25})
    # show plot
    plt.show()

def plotDrugTrafficYear(year):
    df = pd.read_csv('./data/drugs/CosciaRios2012_DataBase.csv')
    df.sort_values('Year', ascending=False, inplace=True)

    G = nx.Graph()

    # first 32 nodes are states, the other 10 are the cartels
    G.add_nodes_from(range(42))

    year_df = df.loc[df['Year'] == year]

    year_df = year_df.iloc[:, [1] + list(range(3, 13))] # get a dataframe for select year only

    for i in range(len(year_df)): # go over every row in the dataframe
        for j in range(1, 11): # families
            # print(year_df.iloc[0,j])
            if year_df.iloc[i, j] == 1:
                G.add_edge(year_df.iloc[i, 0]-1, 31 + j)

    # plot the given year of the network
    pos = nx.spring_layout(G, seed=3113794652) # seed is from 
    colors = ["tab:red", "tab:blue", "tab:green", "tab:cyan", "tab:gray"]

    # nodelists for states and families
    states = list(G.nodes())[0:32]
    families = list(G.nodes())[32:42]

    nx.draw_networkx_nodes(G, pos, nodelist=states, node_color=colors[3], node_size=800)
    nx.draw_networkx_nodes(G, pos, nodelist=families, node_color=colors[4], node_size=800)

    # edges
    #nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    #nx.draw_networkx_edges(G, pos, edge_color=colors[2], width=1.0, alpha=0.5)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    # some math labels
    labels = {}

    for i in range(len(G.nodes())):
        labels[i] = i+1

    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color="whitesmoke")

    plt.tight_layout()
    #plt.title("$\mathcal{L} =$" + str(round(L,2)), fontsize="22")
    plt.axis("off")
    plt.show()

def prepareMalaria():
    # load in the dataset
    # Read in the edgelist, read in attributes as a pandas dataframe, assign node attributes
    G = nx.read_edgelist("./data/malaria/HVR_5.txt", delimiter=',')

    df = pd.read_csv("./data/malaria/metadata_CysPoLV.txt",
                     names=['structure'])
    for node in G.nodes:
        # assuming nodes are indexed from 1 in the original dataset
        G.nodes[node]['structure'] = df['structure'][int(node)-1]

    attr = 'structure'

    return G, attr

def prepareNorwegianBoards():
    # load in the dataset
    # Read in the edgelist, read in attributes as a pandas dataframe, assign node attributes
    G = nx.read_edgelist("./data/norwegian_boards/net1m_2011-08-01.txt")
    df = pd.read_csv("./data/norwegian_boards/data_people.txt", delimiter=" ")

    for node in G.nodes:
        # assuming nodes are indexed from 1 in the original dataset
        G.nodes[node]['gender'] = df['gender'][int(node)-1]

    attr = 'gender'

    return G, attr

def prepareDrugTraffic():
    # this is a temporal graph and it needs some special processing

    df = pd.read_csv('./data/drugs/CosciaRios2012_DataBase.csv')
    df.sort_values('Year', ascending=False, inplace=True)

    G_0_collapsed = nx.Graph()  # years 1-18
    G_1_collapsed = nx.Graph()  # years 1-19
    G_2 = nx.Graph()  # year 19, validation set for level 1
    G_3 = nx.Graph()  # year 20, final testing set

    # first 32 nodes are states, the other 10 are the cartels
    G_0_collapsed.add_nodes_from(range(42))
    G_1_collapsed.add_nodes_from(range(42))
    G_2.add_nodes_from(range(42))
    G_3.add_nodes_from(range(42))

    #temp = []

    # years 2009, 2010
    for x in [[2009, G_2], [2010, G_3]]:
        year_df = df.loc[df['Year'] == x[0]]

        year_df = year_df.iloc[:, [1] + list(range(3, 13))] # get a dataframe for select year only

        for i in range(len(year_df)): # go over every row in the dataframe
            for j in range(1, 11): # families
                if year_df.iloc[i, j] == 1:
                    x[1].add_edge(year_df.iloc[i, 0]-1, 31 + j)

    # years 1990-2008
    for params in [[1990, 2009, G_0_collapsed], [1990, 2010, G_1_collapsed]]:
        for year in range(params[0], params[1]+1):
            year_df = df.loc[df['Year'] == year]
            year_df = year_df.iloc[:, [1] + list(range(3, 13))]

            for i in range(len(year_df)):
                for j in range(1, 11):
                    if year_df.iloc[i, j] == 1:
                        params[2].add_edge(year_df.iloc[i, 0]-1, 31 + j)

    return [G_0_collapsed, G_1_collapsed, G_2, G_3]

def prepareParasites():
    df = pd.read_excel('./data/cold_lake/cold_lake.xls')

    # remove rows and columns we don't need
    df.drop([0,1], inplace=True);
    df.drop(columns=['Unnamed: 0', 'Unnamed: 1', 'Parasite genus', 'Unnamed: 2'], inplace=True)

    hosts = len(df) # number of hosts
    parasites = len(df.columns) # number of parasites

    G = nx.Graph()

    # add hosts, parasites 
    G.add_nodes_from(range(hosts))
    G.add_nodes_from(range(parasites))

    for h in range(hosts):
         for p in range(parasites):
             if df.iloc[h, p] > 0:
                 G.add_edge(h, hosts+p) # add an edge between a host and a parasite if prevalence is nonzero

    return G, ""

def getDatasetAndAttribute(name):
    # simple selector for the datasets I used in my analysis
    # returns desired dataset and one predetermined focal attribute, if it exists
    if name == 'malaria':
        return prepareMalaria()
    elif name == 'norwegian':
        return prepareNorwegianBoards()
    elif name == 'drugs':
        return prepareDrugTraffic()
    elif name == 'parasites':
        return prepareParasites()
    else:
        return None

def getPredictorScoresAndFeatures(G, edges_and_non_edges, attr):
    # runs all implemented algorithms to collect
    # prediction scores and a few topological features
    # for the passed graph

    df = pd.DataFrame()

    # insert edges into the dataframe

    # my goal here was to add as many topological features and prediction results as possible
    # most of the code is either written by me or based on networkx implementations
    # with added multithreading

    jaccard = jaccard_MT(G, edges_and_non_edges)
    shortest_path, attr_score, sp_attr_score = shortest_path_attr_MT(
        G, edges_and_non_edges, attr)
    degree_product_scores = degree_product_MT(G, edges_and_non_edges)
    katz0, katz1 = Katz_centrality_MT(G, edges_and_non_edges)
    adamic_adar = adamic_adar_MT(G, edges_and_non_edges)
    ccpa = common_neighbor_centrality_MT(G, edges_and_non_edges)
    attr_match = attribute_match_MT(G, edges_and_non_edges, attr)
    res_alloc = resource_allocation_index_MT(G, edges_and_non_edges)
    between_centr = betweenness_centrality_MT(G, edges_and_non_edges)

    # this is mostly for posterity, I haven't actually used the pairs for anything I believe
    df['pairs'] = edges_and_non_edges

    # insert features into the dataframe

    df['a'] = np.array(edges_and_non_edges)[:, 0]
    df['b'] = np.array(edges_and_non_edges)[:, 1]
    df['bc_a'] = between_centr[:, 0]
    df['bc_b'] = between_centr[:, 1]
    df['deg_a'] = degree_product_scores[:, 0]
    df['deg_b'] = degree_product_scores[:, 1]
    df['jaccard'] = jaccard
    df['shortest_path'] = shortest_path
    df['degree_product'] = degree_product_scores[:, 2]
    df['katz0'] = katz0
    df['katz1'] = katz1
    df['adamic_adar'] = adamic_adar
    df['ccpa'] = ccpa
    df['res_alloc'] = res_alloc

    features = ['a', 'b', 'bc_a', 'bc_b', 'deg_a', 'deg_b', 'jaccard', 'shortest_path', 'degree_product',
                'katz0', 'katz1', 'adamic_adar', 'ccpa', 'res_alloc']

    if attr != '':
        # we considered a focal attribute
        # so we'll add the features extracted from it to the dataset
        df['attr_match'] = attr_match
        df['attr_score'] = attr_score
        df['sp_attr_score'] = sp_attr_score

        features = features + ['attr_match', 'attr_score', 'sp_attr_score']

    return df, features


def getStackingModelScores(df, label, features, to_erase_edge_ix, to_erase_non_edge_ix):
    # STACKING MODEL IMPLEMENTATION
    # based on:
    # https://www.datacamp.com/tutorial/random-forests-classifier-python

    # X = df[features]  # Features
    # y = df[label]  # Labels

    # training dataset contains only the edges we did not erase
    # as well as a proportional number of non-edges
    # testing contains edges we DID erase and a proportional number of non-edges
    X_train = df.drop(index=np.concatenate(
        (to_erase_edge_ix, to_erase_non_edge_ix)))
    X_test = df.loc[df.index[np.concatenate(
        (to_erase_edge_ix, to_erase_non_edge_ix))]]

    # manually split dataset into training and test
    # shuffle two datasets
    # this seems to help the model a little
    # and I believe this is what train_test_split also does by default
    X_train = X_train.sample(frac=1)
    X_test = X_test.sample(frac=1)

    y_train = X_train[label]
    y_test = X_test[label]

    X_train.drop(['pairs', label], axis=1, inplace=True)
    X_test.drop(['pairs', label], axis=1, inplace=True)

    # Import Random Forest Model
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=1000, max_depth=6, n_jobs=24)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Import scikit-learn metrics module for accuracy calculation
    # Model accuracy computation: this number is REALLY unrepresentative of the actual performance
    # since our graph is sparse and even by saying that no edges exist
    # the model can easily creep towards ~95% accuracy
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # generate probabilities that an edge exists
    predictions = clf.predict_proba(df[features])[0:, 1]

    # add some noise for tie breaking
    predictions = predictions + \
        np.random.normal(0, 0.000001, predictions.shape)

    df['stacking'] = predictions

    return df


def getAUCScores(df, features, T, edges_length, to_erase_edge_ix):
    # get entries for non_edges into a new dataframe

    # all true non_edges
    df_non_edges_only = df.loc[df.index[range(edges_length, len(df))]]
    # all edges that were erased
    df_erased_edges_only = df.loc[df.index[to_erase_edge_ix]]

    # construct a new dataframe for auc computations
    df_for_auc = pd.concat(
        [df_non_edges_only, df_erased_edges_only], ignore_index=True)
    #df_for_auc.set_index(df.columns[0], inplace=True)

    # compute AUC and FPR, TPR
    AUC_vals = AUC_MT(df_for_auc, features, T)

    return AUC_vals


def run(name, roc):
    if name != 'drugs':
        G, attr = getDatasetAndAttribute(name)
        unobserved = 0.5  # fraction of edges that were not observed

        # prepare a reduced graph by removing a fraction of edges
        # number of edges to be removed
        T = int(unobserved * G.number_of_edges())
        G_red = G.copy()  # copy the original graph for edge removal
        edges = list(G.edges())
        non_edges = list(nx.non_edges(G))

        # choose a*m empirical edges for removal and erase them
        to_erase_edge_ix = np.random.choice(
            G_red.number_of_edges(), T, replace=False)

        # used for learning later on -- these non-edges will also be removed from the training set
        to_erase_non_edge_ix = np.random.choice(
            len(non_edges), int(len(non_edges) * unobserved), replace=False) + len(edges)  # add len(edges) to adjust for dataset structure

        to_erase = [edges[ix] for ix in to_erase_edge_ix]
        # erase edges from a copy of the original graph
        G_red.remove_edges_from(to_erase)

    else:
        _, _, G_red, G = getDatasetAndAttribute(name) # use this to predict 2010 based on 2009
        #G_red, G, _, _ = getDatasetAndAttribute(name) # use this to predict 1990-2010 based on 1990-2009
        attr = ''

        edges = list(G.edges())  # edges in next year's network
        non_edges = list(nx.non_edges(G))
        edges_red = list(G_red.edges())  # edges in the current year's network
        # we use this to conceal non-edges from the training set
        unobserved = ((len(edges) - len(edges_red))/(len(edges)))

        # edges which are present in the dataset for next year, but not in current year
        edges_removed = [x for x in edges if x not in edges_red]

        T = len(edges_removed)

        # indices of edges that are in next year, but not in current year
        to_erase_edge_ix = [edges.index(e) for e in edges_removed]

        # used for learning later on -- these non-edges will also be removed from the training set
        to_erase_non_edge_ix = np.random.choice(
            len(non_edges), int(len(non_edges) * unobserved), replace=False) + len(edges)  # add len(edges) to adjust for dataset structure

    # at this point it is easier to lock in the processing order for edges and non-edges
    # by creating a single list with a known structure
    edges_and_non_edges = deepcopy(edges)  # edges are at the top
    edges_and_non_edges.extend(non_edges)  # followed by non-edges

    # retrieve prediction scores and a list of features
    df, features = getPredictorScoresAndFeatures(
        G_red, edges_and_non_edges, attr)

    # create a column for t-vals, recall that we know the exact order they appear in by construction of the dataset
    t = []  # stores actual edge and non-edge flags
    t.extend([1 for _ in range(len(edges))])  # 1 means that the edge exists
    # 0 means that the tuple is a non_edge
    t.extend([0 for _ in range(len(non_edges))])
    df['t'] = t  # insert t column into the dataframe
    label = 't'  # this is the label we'll be training our model to predict

    # a reduced set of features, these are the features we compute AUC for and show
    # on the ROC plot
    features_small = ['jaccard', 'shortest_path', 'degree_product',
                      'adamic_adar', 'ccpa', 'res_alloc', 'stacking']

    #features_small = ['a', 'b', 'bc_a', 'bc_b', 'deg_a', 'deg_b', 'jaccard', 'shortest_path', 'degree_product',
    #           'katz0', 'katz1', 'adamic_adar', 'ccpa', 'res_alloc', 'stacking']

    # update the dataframe to contain stacking model scores
    df = getStackingModelScores(
        df, label, features, to_erase_edge_ix, to_erase_non_edge_ix)
    # stacking will count as a new predictor for AUC computation
    features.append("stacking")

    AUC_vals = getAUCScores(df, features_small, T,
                            len(edges), to_erase_edge_ix)

    # print out AUC vals for considered features
    #for feat in features_small:
    #    print(feat + ': ' + str(AUC_vals[feat][0]))

    if roc == True:
        #plot ROC
        plotROC(features_small, AUC_vals)

    # plot predicted links for the drugs network
    if name == 'drugs' and roc == True:
        # this is temporarily the same code as in AUC because I need the same dataframe

        # all true non_edges
        df_non_edges_only = df.loc[df.index[range(len(edges), len(df))]]
        # all edges that were erased
        df_erased_edges_only = df.loc[df.index[to_erase_edge_ix]]

        # construct a new dataframe for auc computations
        df_for_auc = pd.concat(
            [df_non_edges_only, df_erased_edges_only], ignore_index=True)
        #df_for_auc.set_index(df.columns[0], inplace=True)

        df_for_auc.sort_values('stacking', ascending=False, inplace=True) # sort the dataframe by stacking score
        edges_recovered = df_for_auc['pairs'].to_numpy()[0 : len(edges_removed)*2] # 

        #visualizeDrugNetwork(G_red, edges_removed)
        visualizeDrugNetwork(G, edges_removed, list(edges_recovered))

    return AUC_vals, features_small

def multiRun(name, n):
    # run the experiment multiple times, record AUC, plot the boxplot

    AUC_multirun_data = []
    features = None
    for i in range(n):
        auc_vals, features_ = run(name, False)

        AUC_multirun_data.append([auc_vals[feat][0] for feat in features_])

        if features == None:
            features = features_

        print("Iterations complete:\t" + str(i+1))

    plotBoxPlot(features, np.array(AUC_multirun_data))

### run once and plot roc, for drugs network also plot predictions
# run ('malaria', True)
# run ('norwegian', True)
# run('drugs', True)
run('parasites', True)

### auc box plots
# multiRun('malaria', 50)
# multiRun('norwegian', 50)
# multiRun('drugs', 50)
# multiRun('parasites', 50)
