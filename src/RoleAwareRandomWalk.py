"""RoleBased2Vec Machine."""

import math
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import random
from sklearn.cluster import KMeans
from motif_count import MotifCounterMachine
from utils import  get_roles_nodes, load_graph
from weisfeiler_lehman_labeling import WeisfeilerLehmanMachine
from gensim.models import Word2Vec
from graphrole import RecursiveFeatureExtractor, RoleExtractor
def join_strings(features):
    """
    Creating string labels by joining the individual quantile labels.
    """
    return {str(node): ["_".join(features[node])] for node in features} #str(node)

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
    km = KMeans(init='k-means++',n_clusters = n, n_init=10, random_state=42)#n_jobs = -1
    model = km.fit(vector)
    #labels = model.predict(vector)
    labels = model.labels_
    features = {str(sorted(data.keys())[node]): [str(labels[node])] for node in range(len(data))}
    print('finish KMeans')
    return features

class RoleBased2Vec():

    def __init__(self, args,G, r, t, num_walks, walk_length,window_size): #

        self.args = args
        self.G = G
        self.is_directed = False
        self.r = r
        self.t = t
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.window_size = window_size
        # self.d =d

    def create_graph_structural_features(self,graph):
        """
        Extracting structural features.
        """
        if self.args.features == "wl":
            print('We are using WL...')
            features = {str(node): str(int(math.log(graph.degree(node)+1, self.args.log_base))) for node in graph.nodes()}
            machine = WeisfeilerLehmanMachine(graph, features, self.args.labeling_iterations)
            machine.do_recursions()
            features = machine.extracted_features
            # features = join_strings(features)
        elif self.args.features == 'motif':
            machine = MotifCounterMachine(graph, self.args)
            features = machine.create_string_labels()
        elif self.args.features == 'motif_tri':
            print('We are using motif_tri...,k = {}'.format(self.args.clusters))
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

        else:
            print('We are using degree...')
            features = {str(node): [str(graph.degree(node))] for node in graph.nodes()}
            # features = {str(node): str(int(math.log(graph.degree(node)+1, self.args.log_base))) for node in graph.nodes()}
        return features



    def walk_step(self,v):
        nbs = list(self.G.neighbors(v))
        role_list = self.roles_nodes[self.structura_features[str(v)][0]]

        all_nbs = nbs+ role_list
        weight_1 = [1]*len(nbs)
        weight_2 = [1]*len(role_list)
        for i,x in enumerate(nbs):
            weight_1[i] = 1/self.r
        for i,x in enumerate(role_list):
            weight_2[i] = 1/self.t

        weights = weight_1+weight_2
        # norm_const = sum(weights)
        # weights = [float(u_prob)/norm_const for u_prob in weights]
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

    def sentenses(self,is_load_feature = False):
        from tqdm import tqdm
        if is_load_feature:
            print('loading the structural features........')
            self.structura_features = np.load(self.args.output + self.args.dataset + 'structure_features' + '.npy',
                                              allow_pickle=True).item()
            self.roles_nodes = get_roles_nodes(self.structura_features)
        else:
            self.structura_features = self.create_graph_structural_features(self.G)
            self.roles_nodes = get_roles_nodes(self.structura_features)

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

def draw_role_list(data):
    import matplotlib.pyplot as plt
    d = {}
    for items in data.items():
        d[int(items[0])] = len(list(items[1]))

    fig, ax = plt.subplots()
    ax.bar(d.keys(), d.values())
    plt.show()
