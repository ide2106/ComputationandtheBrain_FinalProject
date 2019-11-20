import networkx as nx
import itertools
from networkx import graphml
import numpy as np
import matplotlib.pyplot as plt

def main():

    graphs = []

    graph1 = nx.MultiDiGraph()
    graph1.add_node('n0')
    graph1.add_node('n1')
    graph1.add_node('n2')
    graphs.append(graph1)

    graph2 = nx.MultiDiGraph()
    graph2.add_node('n0')
    graph2.add_node('n1')
    graph2.add_node('n2')
    graph2.add_edge('n1','n0')
    graphs.append(graph2)

    graph3 = nx.MultiDiGraph()
    graph3.add_node('n0')
    graph3.add_node('n1')
    graph3.add_node('n2')
    graph3.add_edge('n1','n2')
    graph3.add_edge('n1','n0')
    graphs.append(graph3)

    graph4 = nx.MultiDiGraph()
    graph4.add_node('n0')
    graph4.add_node('n1')
    graph4.add_node('n2')
    graph4.add_edge('n2','n0')
    graph4.add_edge('n1','n0')
    graphs.append(graph4)

    graph5 = nx.MultiDiGraph()
    graph5.add_node('n0')
    graph5.add_node('n1')
    graph5.add_node('n2')
    graph5.add_edge('n0','n2')
    graph5.add_edge('n1','n0')

    graphs.append(graph5)

    graph6 = nx.MultiDiGraph()
    graph6.add_node('n0')
    graph6.add_node('n1')
    graph6.add_node('n2')
    graph6.add_edge('n0','n1')
    graph6.add_edge('n1','n0')

    graphs.append(graph6)

    graph7 = nx.MultiDiGraph()
    graph7.add_node('n0')
    graph7.add_node('n1')
    graph7.add_node('n2')
    graph7.add_edge('n0','n2')
    graph7.add_edge('n1','n0')
    graph7.add_edge('n2','n0')


    graphs.append(graph7)

    graph8 = nx.MultiDiGraph()
    graph8.add_node('n0')
    graph8.add_node('n1')
    graph8.add_node('n2')
    graph8.add_edge('n0','n2')
    graph8.add_edge('n0','n1')
    graph8.add_edge('n2','n0')


    graphs.append(graph8)

    graph9 = nx.MultiDiGraph()
    graph9.add_node('n0')
    graph9.add_node('n1')
    graph9.add_node('n2')
    graph9.add_edge('n0','n1')
    graph9.add_edge('n0','n2')
    graph9.add_edge('n1','n0')
    graph9.add_edge('n2','n0')


    graphs.append(graph9)

    graph10 = nx.MultiDiGraph()
    graph10.add_node('n0')
    graph10.add_node('n1')
    graph10.add_node('n2')
    graph10.add_edge('n1','n2')
    graph10.add_edge('n0','n1')
    graph10.add_edge('n0','n2')



    graphs.append(graph10)

    graph11 = nx.MultiDiGraph()
    graph11.add_node('n0')
    graph11.add_node('n1')
    graph11.add_node('n2')
    graph11.add_edge('n1','n2')
    graph11.add_edge('n0','n1')
    graph11.add_edge('n2','n0')



    graphs.append(graph11)

    graph12 = nx.MultiDiGraph()
    graph12.add_node('n0')
    graph12.add_node('n1')
    graph12.add_node('n2')
    graph12.add_edge('n1','n0')
    graph12.add_edge('n2','n0')
    graph12.add_edge('n1','n2')
    graph12.add_edge('n0','n2')


    graphs.append(graph12)

    graph13 = nx.MultiDiGraph()
    graph13.add_node('n0')
    graph13.add_node('n1')
    graph13.add_node('n2')
    graph13.add_edge('n0','n1')
    graph13.add_edge('n2','n0')
    graph13.add_edge('n1','n2')
    graph13.add_edge('n0','n2')



    graphs.append(graph13)

    graph14 = nx.MultiDiGraph()
    graph14.add_node('n0')
    graph14.add_node('n1')
    graph14.add_node('n2')
    graph14.add_edge('n0','n1')
    graph14.add_edge('n2','n0')
    graph14.add_edge('n2','n1')
    graph14.add_edge('n0','n2')




    graphs.append(graph14)

    graph15 = nx.MultiDiGraph()
    graph15.add_node('n0')
    graph15.add_node('n1')
    graph15.add_node('n2')
    graph15.add_edge('n0','n1')
    graph15.add_edge('n1','n0')
    graph15.add_edge('n2','n1')
    graph15.add_edge('n2','n0')
    graph15.add_edge('n2', 'n0')
    graph15.add_edge('n0','n2')



    graphs.append(graph15)

    graph16 = nx.MultiDiGraph()
    graph16.add_node('n0')
    graph16.add_node('n1')
    graph16.add_node('n2')
    graph16.add_edge('n0','n1')
    graph16.add_edge('n1','n0')
    graph16.add_edge('n2','n1')
    graph16.add_edge('n0','n2')
    graph16.add_edge('n2','n0')
    graph16.add_edge('n1','n2')



    graphs.append(graph16)

    for graph in graphs:
        graph_connections(graph, 3, 'test')


def load_graph(filename):
    graph = graphml.read_graphml(filename)
    return graph

def graph_connections(graph, group_size, filename):
    connections = {}
    for i in range(1,17):
        connections[i] = 0
    #remove self-loops
    self_edges = []
    for e in graph.edges():
        if e[0] == e[1]:
            self_edges.append(e)
            print('SELFEDGE')
    for e in self_edges:
        graph.remove_edge(e[0],e[0])

    nodes = graph.nodes()
    subsets = itertools.combinations(nodes, group_size)

    for s in subsets:
        subset = nx.DiGraph(graph.subgraph(s))

        edges = list(subset.edges())

        if subset.size() == 0:
            connections[1] += 1
            continue
        elif subset.size() == 1:
            connections[2] += 1
            continue
        elif subset.size() == 5:
            connections[15] += 1
            continue
        elif subset.size() == 6:
            connections[16] += 1
            continue
        elif subset.size() == 2:
            if edges[0][0] == edges[1][0]:
                connections[3] += 1
            elif edges[0][1] == edges[1][1]:
                connections[4] += 1
            elif edges[0][0] != edges[1][1] or edges[0][1] != edges[1][0]:
                connections[5] += 1
            else:
                connections[6] += 1
            continue
        elif subset.size() == 3:
            cluster_type = None
            for i in range(0,3):
                if (edges[i][0] != edges[(i+1)%3][0] and edges[i][0] != edges[(i+2)%3][0]
                        and edges[i][0] != edges[(i+1)%3][1] and edges[i][0] != edges[(i+2)%3][1]):
                    cluster_type = 7
                    break
                if (edges[i][1] != edges[(i+1)%3][1] and edges[i][1] != edges[(i+2)%3][1]
                    and edges[i][1] != edges[(i+1)%3][0] and edges[i][1] != edges[(i+2)%3][0]):
                    cluster_type = 8
                    break
                if ((edges[i][0] == edges[(i+1)%3][0] and edges[i][0] != edges[(i+2)%3][1])
                    or (edges[i][0] == edges[(i+2)%3][0] and edges[i][0] != edges[(i+1)%3][1])):
                    cluster_type = 10
                    break
                #if (edges[i][0] != edges[(i+1)%3][0] and edges[i][0] != edges[(i+2)%3][0] and
                 #   edges[(i+1)%3][0] != edges[(i+2)%3][0]):
                  #  cluster_type = 11
                   # break
            if cluster_type is not None:
                connections[cluster_type] += 1
            else:
                connections[11] +=1
        elif subset.size() == 4:
            cluster_type = None
            for i in range(0, 4):
                if ((edges[i][0] == edges [(i+1)%4][0] or edges[i][0] == edges [(i+2)%4][0] or edges[i][0] == edges [(i+3)%4][0]) and
                    edges[i][0] != edges [(i+1)%4][1] and edges[i][0] != edges [(i+2)%4][1] and edges[i][0] != edges [(i+3)%4][1]):
                    cluster_type = 12
                    break
                if ((edges[i][1] == edges [(i+1)%4][1] or edges[i][1] == edges [(i+2)%4][1] or edges[i][1] == edges [(i+3)%4][1]) and
                    edges[i][1] != edges [(i+1)%4][0] and edges[i][1] != edges [(i+2)%4][0] and edges[i][1] != edges [(i+3)%4][0]):
                    cluster_type = 14
                    break
                if ((edges[i][0] != edges[(i+1)%4][0] and (edges[i][0], edges[(i+1)%4][0]) not in edges and (edges[(i+1)%4][0], edges[i][0]) not in edges) or
                    (edges[i][0] != edges[(i+2)%4][0] and (edges[i][0], edges[(i+2)%4][0]) not in edges and (edges[(i+2)%4][0], edges[i][0]) not in edges) or
                    (edges[i][0] != edges[(i+3)%4][0] and (edges[i][0], edges[(i+3)%4][0]) not in edges and (edges[(i+3)%4][0], edges[i][0]) not in edges)):
                        cluster_type = 9

            if cluster_type is not None:
                connections[cluster_type] += 1
            else:
                cluster_type = 13
                connections[cluster_type] += 1
            continue

    print(connections)
    return


main()