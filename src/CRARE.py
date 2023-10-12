"""example of CRARE method"""

from param_parser import parameter_parser
from utils import tab_printer
from C_RoleAwareRandomWalk import RoleBased2Vec
import networkx as nx
import numpy as np
import pandas as pd
import random

def read_graph(graph_path, is_directed=False):
    data = np.loadtxt(graph_path, dtype=int)

    data = data.astype(int)
    edgelist1 = []  # 1 advice

    for i in range(len(data)):
        edgelist1.append((data[i][0], data[i][1]))

    if is_directed == True:
        G1 = nx.DiGraph()
    else:
        G1 = nx.Graph()
    G1.add_edges_from(edgelist1)
    G1.remove_edges_from(nx.selfloop_edges(G1))

    # G1.remove_edges_from(G1.selfloop_edges())

    return G1

def get_node_embedding(args,G,r,t,m):
    import time
    start0 = time.time()
    print('get embedding time:')
    model = RoleBased2Vec(args, G, r,t,m, num_walks=10, walk_length=80, window_size=10)
    # w2v = model.create_embedding()
    w2v = model.train(workers=4)
    print('******************************embedding时间：{}***********************'.format(time.time() - start0))
    return w2v.wv

if __name__ == "__main__":
    #example
    path = r'../data/Edgelist/brazil-airports.edgelist'  #
    label_path = r'../data/Category/labels-brazil-airports.txt'

    args = parameter_parser()
    tab_printer(args)
    G = read_graph(path)
    # node embedding
    X = get_node_embedding(args, G, r=4,t=0.25,m=1)
    # print(X)
