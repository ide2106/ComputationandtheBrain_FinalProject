import networkx
import random
import math
import numpy as np

# set these to use main to generate an ROC graph
S = 20   # community size
Q = 0.3  # community connectivity
B = 0    # 0 = not bipartite, else bipartite communities


printing = False


def main():

    n = 400
    p = 0.10956967359191465

    # some out of the box random graphs I tested things on earlier

    # fast_erdos = networkx.fast_gnp_random_graph(n,p, directed=True)
    # gnp = networkx.gnp_random_graph(n, p, directed=True)
    erdos_reyi = networkx.erdos_renyi_graph(n, p, directed=True)
    # gnm = networkx.gnm_random_graph(n,m, directed=True)
    # newman = networkx.newman_watts_strogatz_graph(n, k, p)
    # watts = networkx.newman_watts_strogatz_graph(n, k, p)
    # powerlaw = networkx.random_powerlaw_tree(n)

    networkx.write_graphml(erdos_reyi, 'erdos_reyi.graphml')

    # ROC graphs I generated to test function
    dist = [(S, Q, B)]
    roc_graph = roc_generation(dist, 200, 0.2)
    networkx.write_graphml(roc_graph, 'roc_graph.graphml')

    dist = [(S, Q, 1)]
    roc_graph = roc_generation(dist, 200, 0.2)
    networkx.write_graphml(roc_graph, 'roc_graph_bipartite.graphml')

    dist = [(200, 0.2 / 200, 0)]
    roc_graph = roc_generation(dist, 200, 0.2)
    networkx.write_graphml(roc_graph, 'roc_graph_erdos.graphml')


def roc_generation(distribution, n, d):
    """This function takes a distribution of (s,q,b) tuples
    (currently as a list, I previously tried normal distribution over ranges for s and q and 0, 1 probabilities for p)
    It generates an ROC graph for the distribution, n and d"""

    graph = networkx.DiGraph()
    graph.add_nodes_from(range(0, n))

    edges = 0
    max_edges = math.factorial(n) / math.factorial(n - 2)

    # continuously chooses a community from distribution and adds edges accordingly, until number of edges is met
    while edges < d*max_edges:
        s, q, b = get_triple2(distribution)
        community_nodes = []
        for node in graph:
            r = random.random()
            if r < s/n:
                community_nodes.append(node)

        if b == 0:  # adds edges randomly in whole community
            community_graph = networkx.erdos_renyi_graph(len(community_nodes), q, directed=True)
            community_graph = networkx.convert_node_labels_to_integers(community_graph)
            for edge in community_graph.edges():
                graph.add_edge(community_nodes[edge[0]], community_nodes[edge[1]])
                edges += 1
        else:  # makes edges in bipartite subgraphs for community
            if printing:
                print()
                print(b, len(community_nodes)-b, q)
                print()
            community_graph = networkx.algorithms.bipartite.random_graph(b, len(community_nodes)-b, q, directed=True)
            community_graph = networkx.convert_node_labels_to_integers(community_graph)
            for edge in community_graph.edges():
                graph.add_edge(community_nodes[edge[0]], community_nodes[edge[1]])
                edges += 1

    return graph


def get_triple(dist):
    """This function takes a distribution and returns a random community in it
        This version is for distributions as two normal probability distributions and a b=0 probability
        distributions are given as mean and standard deviation, either 1 or a list"""
    s_dist = dist[0]
    q_dist = dist[1]
    p_b = dist[2]
    s = np.random.normal(s_dist[0], s_dist[1])

    if len(s) > 1:  # if multiple options are given, pick one uniformly at random
        r = random.randint(0, len(s)-1)
        s = s[r]
    q = np.random.normal(q_dist[0], q_dist[1])

    if len(q) > 1:
        r = random.randint(0, len(q)-1)
        q = q[r]

    r = random.random()
    if r > p_b:
        b = s // 2
    else:
        b = 0

    return s, q, b


def get_triple2(dist):
    """This function takes a distribution and returns a random community in it
    This version is for distributions as lists of possible communities
    Not a great way to represent distributions, but testing with the other was computationally impossible for me"""

    r = random.randint(0, len(dist)-1)
    s, q, b = dist[r]
    if b != 0:
        b = s // 2
    return s, q, b

