"""RoleBased2Vec Machine."""

import math
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import random
from motif_count import MotifCounterMachine
from utils import  get_roles_nodes, load_graph,get_community_nodes
from weisfeiler_lehman_labeling import WeisfeilerLehmanMachine
from gensim.models import Word2Vec
import community as community_louvain
from sklearn.cluster import KMeans

def get_feature_vec(data):
    n_tot = len(data)
    node = random.choice(list(data.keys()))
    dimensions = len(data[node])
    feature_vec = np.empty((n_tot, dimensions), dtype='f') #
    for ii in range(n_tot):
        v = sorted(data.keys())[ii]
        feature_vec[ii] = [ int(i) for i in data[v]]
    return feature_vec

def KM_train(data,n=10):
    vector = get_feature_vec(data)
    print('start KMeans')
    km = KMeans(n_clusters = n,random_state=42)#n_jobs = -1
    model = km.fit(vector)
    #labels = model.predict(vector)
    labels = model.labels_
    features = {str(sorted(data.keys())[node]): [str(labels[node])] for node in range(len(data))}
    print('finish KMeans')
    return features

class RoleBased2Vec():
    """
    multi_RoleBased2Vec model class.
    """
    def __init__(self, args,G, r, t, m,num_walks, walk_length, window_size): #

        self.args = args
        self.G = G
        self.is_directed = False
        self.r = r
        self.t = t
        self.m = m
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.window_size = window_size
        # self.d = d

    def create_graph_structural_features(self,graph):
        """
        Extracting structural features.
        """
        if self.args.features == "wl":
            features = {str(node): str(int(math.log(graph.degree(node)+1, self.args.log_base))) for node in graph.nodes()}
            machine = WeisfeilerLehmanMachine(graph, features, self.args.labeling_iterations)
            machine.do_recursions()
            features = machine.extracted_features
        elif self.args.features == 'motif':
            machine = MotifCounterMachine(graph, self.args)
            features = machine.create_string_labels()
        elif self.args.features == 'motif_tri':
            features_d = {str(node): [str(int(math.log(graph.degree(node)+1, self.args.log_base)))] for node in graph.nodes()}
            # features_d = {str(node): [str(graph.degree(node))] for node in graph.nodes()}
            tr = nx.triangles(graph)
            features_tr = {str(node): str(int(math.log(tr[node] + 1, self.args.log_base))) for node in tr}
            # features_tr = {str(node): str(tr[node]) for node in tr}
            for node in features_d:
                features_d[node].append(features_tr[node])
            ###string feature
            # features = features_d #join_strings(features_d)

            features = KM_train(features_d,n=self.args.clusters)

        else:#if self.args.features == "degree":
            features = {str(node): [str(graph.degree(node))] for node in graph.nodes()}
        # np.save(self.args.output + self.args.dataset + 'structure_features' +'.npy', features, allow_pickle=True)
        return features

    def get_graph_community(self,G):
        """
        Extracting structural features.
        """
        partition = community_louvain.best_partition(G)
        return partition

    def walk_step(self,v):
        nbs = list(self.G.neighbors(v))
        role_list = self.roles_nodes[self.structura_features[str(v)][0]]
        c_list = self.community_nodes[self.community_features[v]]

        all_nbs = nbs+ role_list +c_list

        weight_1 = [1]*len(nbs)
        weight_2 = [1]*len(role_list)
        weight_3 = [1]*len(c_list)
        for i,x in enumerate(nbs):
            weight_1[i] = 1/self.r
        for i,x in enumerate(role_list):
            weight_2[i] = 1/self.t
        for i,x in enumerate(c_list):
            weight_3[i] = 1/self.m

        weights = weight_1+weight_2 + weight_3

        return random.choices(all_nbs, weights=weights, k=1)[0]

    def random_walk(self):
        # random walk with every node as start point
        walks = []
        for node in self.G.nodes():
            walk = [node]
            nbs = list(self.G.neighbors(node))
            #####
            # role_list = self.roles_nodes[self.structura_features[str(node)][0]]
            # all_nbs = nbs + role_list
            # if len(all_nbs)>0:
            if len(nbs) > 0:
                walk.append(random.choice(nbs))
                for i in range(2, self.walk_length):
                    v = self.walk_step(walk[-1])
                    if not v:
                        break
                    walk.append(v)
            walk = [str(x) for x in walk]
            walks.append(walk)

        return walks

    def sentenses(self):
        from tqdm import tqdm
        self.structura_features = self.create_graph_structural_features(self.G)
        self.roles_nodes = get_roles_nodes(self.structura_features)
        self.community_features = self.get_graph_community(self.G)  # {node: community}
        self.community_nodes = get_community_nodes(self.community_features)  # {community: [nodes]}

        sts = []
        for _ in tqdm(range(self.num_walks)):
            sts.extend(self.random_walk())
        return sts

    def save_walks(self,walks):
        """
        Save node2vec walks.
        """
        out_file = self.args.output + self.args.dataset + 'walks' +'.txt'
        with open(out_file, "w") as f_out:
            for walk in walks:
                f_out.write(" ".join(map(str, walk)) + "\n")
            # print("Elapsed time during walks: ", elapsed, " seconds.\n")
        return

    def train(self, workers=4):

        print('Random walk to get training data...')
        sentenses = self.sentenses()
        self.save_walks(sentenses)
        print('Number of sentenses to train: ', len(sentenses))

        # with open(self.args.output + self.args.dataset+'walk_RoleBased2vec.txt', 'w') as f:
        #     f.write(str(sentenses))

        print('Start training...')
        random.seed(616)
        w2v = Word2Vec(sentences=sentenses, size=self.args.dimensions, window=self.window_size, iter=self.args.num_iters, sg=1,
                       hs=1, min_count=0, workers=workers)

        # w2v.save(self.args.output + self.args.dataset +'RoleBased2Vec.model')
        print('Training Done.')

        return w2v


from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

def create_documents(features):
    """
    Created tagged documents object from a dictionary.
    :param features: Keys are document ids and values are strings of the document.
    :return docs: List of tagged documents.
    """
    docs = [TaggedDocument(words=v, tags=[str(k)]) for k, v in features.items()]
    return docs

