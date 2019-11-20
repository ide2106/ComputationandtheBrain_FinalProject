import networkx

def main():
    n = 400
    m = 30
    k = 10
    p = 0.10956967359191465
    p1 = .9
    p2 = .2
    d = 1
    #fast_erdos = networkx.fast_gnp_random_graph(n,p, directed=True)
    gnp = networkx.gnp_random_graph(n, p, directed=True)
    erdos_reyi = networkx.erdos_renyi_graph(n,p, directed=True)
    #erdos_binomial = networkx.binomial_graph(n,p, directed=True)
    #dense_gnm = networkx.dense_gnm_random_graph(n,m)
    #gnm = networkx.gnm_random_graph(n,m, directed=True)
    #newman = networkx.newman_watts_strogatz_graph(n, k, p)
    #watts = networkx.newman_watts_strogatz_graph(n, k, p)
    #connected_watts = networkx.connected_watts_strogatz_graph(n, k, p)
    #regular = networkx.random_regular_graph(d, n)
    duplication = networkx.duplication_divergence_graph(n,p)
    #lobster = networkx.random_lobster(n, p1, p2)
    #shell = networkx.random_shell_graph() ?
    #powerlaw = networkx.random_powerlaw_tree(n)
    #powerlaw_seq = networkx.random_powerlaw_tree_sequence(n)

    networkx.write_graphml(gnp, 'gnp_graph.graphml')
    networkx.write_graphml(erdos_reyi, 'erdos_reyi.graphml')



main()