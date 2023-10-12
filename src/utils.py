"""Dataset utilities and printing."""

import pandas as pd
import numpy as np
import networkx as nx
from texttable import Texttable
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def load_graph(graph_path):

    graph = nx.from_edgelist(pd.read_csv(graph_path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def read_graph(input, enforce_connectivity=False,weighted=False, directed=False):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()
        # Take largest connected subgraph
        if enforce_connectivity and not nx.is_connected(G):
            G = max(nx.connected_component_subgraphs(G), key=len)
            print("Input graph not connected: using largest connected subgraph")

    # Remove nodes with self-edges
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from([v for v in nx.isolates(G)])
    print("Read graph, nodes: %d, edges: %d" % (G.number_of_nodes(), G.number_of_edges()))

    return G

def get_roles_nodes(data):

    # if set(data.values()):
    #     roles_nodes = {role: [] for role in set(data.values())}
    # else:
    ###string
    value_list = set([i[0] for i in data.values()])
    roles_nodes = {role: [] for role in value_list}

    for role in value_list:
        for node in data:
            if data[str(node)][0] == role:
            # if data[node][0] == role:
                roles_nodes[role].append(int(node))
    return roles_nodes

def get_community_nodes(data):

    # if set(data.values()):
    #     roles_nodes = {role: [] for role in set(data.values())}
    # else:
    ###string 方式
    value_list = set([i for i in data.values()])
    c_nodes = {role: [] for role in value_list}

    for role in value_list:
        for node in data:
            if data[node] == role:
                c_nodes[role].append(int(node))
    return c_nodes

def create_documents(features):
    """
    Created tagged documents object from a dictionary.
    :param features: Keys are document ids and values are strings of the document.
    :return docs: List of tagged documents.
    """
    docs = [TaggedDocument(words=v, tags=[str(k)]) for k, v in features.items()]
    return docs

#