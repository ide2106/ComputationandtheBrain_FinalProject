from pathlib import Path
import os
import networkx as nx
import itertools
from networkx import graphml
import math
import numpy as np
import matplotlib.pyplot as plt

# folders to run on
CONNECTOMEGRAPHFILE = 'ConnectomeGraphs'
GENERATEDGRAPHFILE = 'GeneratedGraphs'
ROCGRAPHFILE = 'RocGraphs'

# control variables - print statements, and graphs as popups vs saved in files
printing = False
show_graphs = False
save_graphs = True


def main():
    """ This runs the connection analysis on all .graphml files in the chosen folder.
     Set the folder with the graph_set variables using options above.
     Note: main is not currently called in this file, add to bottom to run. """

    # chose while folder of graphs to analyze
    graph_set = CONNECTOMEGRAPHFILE

    for file in os.listdir(Path.cwd() / graph_set):
        if file.endswith(".graphml"):
            output_file = 'data_' + file + '.txt'
            fileout = open(output_file, 'w')
            graph = graphml.read_graphml(graph_set + '\\' + file)
            connections_dict = get_connections(graph, fileout)
            probabilities_dict = get_probabilities(graph, fileout)
            barchart(connections_dict, probabilities_dict, file, fileout)


def compare_vs_rat():
    """ Similar to main, compares graphs to the Rat graph probabilities rather than random ones
    It can be used to check many graphs against a target graph (I used against ROC graph folder)
    To compare to a different graph, you need to copy the connection type dictionary
    from the output.txt file for the target graph """

    # chose while folder of graphs to analyze
    graph_set = CONNECTOMEGRAPHFILE

    for file in os.listdir(Path.cwd() / graph_set):
        if file.endswith(".graphml"):
            rat_output_file = 'data_' + file + 'vsRat.txt'
            ratfileout = open(rat_output_file, 'w')
            graph = graphml.read_graphml(graph_set + '\\' + file)
            connections_dict = get_connections(graph, ratfileout)
            rat_data = {1: 0.6934838235420362, 2: 0.07870262974956996, 3: 0.0058078895000823125, 4: 0.14106704573000958,
                        5: 0.0010064858362765648, 6: 0.014418771622477839, 7: 0.009029203835602222,
                        8: 0.0009853326067878817, 9: 0.028175247961144078, 10: 0.006974257705431414,
                        11: 8.252605226526663e-06, 12: 0.011328929825394319, 13: 0.00019692423506056724,
                        14: 0.0011571195960435113, 15: 0.006257514198631006, 16: 0.0014005714502260478}
            barchart(connections_dict, rat_data, file + 'vsRat', ratfileout)


def get_connections(graph, file):
    """This is the important function
    It takes a graph and calculates the occurance of each 3 way connection type
    It outputs information to the given file, as well as printing if that is turned on"""

    connections = {}  # to save number of each type of connection, types kept same as paper
    for i in range(1, 17):  # numbers are also from paper, I didn't make the indexing
        connections[i] = 0

    self_edges = []  # remove self-loops
    for e in graph.edges():
        if e[0] == e[1]:
            self_edges.append(e)
    for e in self_edges:
        graph.remove_edge(e[0], e[0])

    nodes = graph.nodes()
    if printing:
        print(str(len(nodes)) + ' nodes')
    file.write(str(len(nodes)) + ' nodes')
    file.write('\n\n')

    subsets = itertools.combinations(nodes, 3)

    # loop through each of the n choose 3 subsets and categorize type
    counter = 0
    for s in subsets:
        counter += 1
        subset = nx.DiGraph(graph.subgraph(s))  # this removes duplicate edges
        edges = list(subset.edges())

        if len(edges) == 0:
            connections[1] += 1

        elif len(edges) == 1:
            connections[2] += 1

        elif len(edges) == 5:
            connections[15] += 1

        elif len(edges) == 6:
            connections[16] += 1

        elif len(edges) == 2:
            if edges[0][0] == edges[1][0]:
                connections[3] += 1
            elif edges[0][1] == edges[1][1]:
                connections[4] += 1
            elif edges[0][0] != edges[1][1] or edges[0][1] != edges[1][0]:
                connections[5] += 1
            else:
                connections[6] += 1

        elif len(edges) == 3:
            cluster_type = None
            for i in range(0, 3):
                if (edges[i][0] != edges[(i + 1) % 3][0] and edges[i][0] != edges[(i + 2) % 3][0]
                        and edges[i][0] != edges[(i + 1) % 3][1] and edges[i][0] != edges[(i + 2) % 3][1]):
                    cluster_type = 7
                    break
                if (edges[i][1] != edges[(i + 1) % 3][1] and edges[i][1] != edges[(i + 2) % 3][1]
                        and edges[i][1] != edges[(i + 1) % 3][0] and edges[i][1] != edges[(i + 2) % 3][0]):
                    cluster_type = 8
                    break
                if ((edges[i][0] == edges[(i + 1) % 3][0] and edges[i][0] != edges[(i + 2) % 3][1])
                        or (edges[i][0] == edges[(i + 2) % 3][0] and edges[i][0] != edges[(i + 1) % 3][1])):
                    cluster_type = 10
                    break
            if cluster_type is not None:
                connections[cluster_type] += 1
            else:
                connections[11] += 1
        elif len(edges) == 4:
            # if you are looking at this and want proof it works,
            # generate a regular Erdos-Renyi graph with graph_generation and run it. I know it is messy.
            cluster_type = None
            for i in range(0, 4):
                if ((edges[i][0] == edges[(i + 1) % 4][0] or edges[i][0] == edges[(i + 2) % 4][0] or edges[i][0] ==
                     edges[(i + 3) % 4][0]) and
                        edges[i][0] != edges[(i + 1) % 4][1] and edges[i][0] != edges[(i + 2) % 4][1] and edges[i][0] !=
                        edges[(i + 3) % 4][1]):
                    cluster_type = 12
                    break
                if ((edges[i][1] == edges[(i + 1) % 4][1] or edges[i][1] == edges[(i + 2) % 4][1] or edges[i][1] ==
                     edges[(i + 3) % 4][1]) and
                        edges[i][1] != edges[(i + 1) % 4][0] and edges[i][1] != edges[(i + 2) % 4][0] and edges[i][1] !=
                        edges[(i + 3) % 4][0]):
                    cluster_type = 14
                    break
                if ((edges[i][0] != edges[(i + 1) % 4][0] and (edges[i][0], edges[(i + 1) % 4][0]) not in edges and (
                        edges[(i + 1) % 4][0], edges[i][0]) not in edges) or
                        (edges[i][0] != edges[(i + 2) % 4][0] and (
                                edges[i][0], edges[(i + 2) % 4][0]) not in edges and (
                                 edges[(i + 2) % 4][0], edges[i][0]) not in edges) or
                        (edges[i][0] != edges[(i + 3) % 4][0] and (
                                edges[i][0], edges[(i + 3) % 4][0]) not in edges and (
                                 edges[(i + 3) % 4][0], edges[i][0]) not in edges)):
                    cluster_type = 9
                    break

            if cluster_type is not None:
                connections[cluster_type] += 1
            else:
                cluster_type = 13
                connections[cluster_type] += 1
    if printing:
        print(str(counter) + ' subsets examined')
        print(connections)
        check = 0
        for k in connections.keys():
            check += connections[k]
        print(str(check) + ' subsets graphed')

    return connections


def get_probabilities(graph, file):
    """This takes a graph and outputs the probability of each 3 way connection type if the graph was random"""

    self_edges = []  # remove self-loops
    for e in graph.edges():
        if e[0] == e[1]:
            self_edges.append(e)
    for e in self_edges:
        graph.remove_edge(e[0], e[0])

    graph = nx.DiGraph(graph)  # remove duplicate edges

    node_count = len(graph.nodes)
    edge_count = len(graph.edges)

    max_edges = math.factorial(node_count) / math.factorial(node_count - 2)
    p = edge_count / max_edges
    if printing:
        print('p is ' + str(p))
    file.write('p is ' + str(p))
    file.write('\n\n')
    probabilities = {1: (1 - p) ** 6, 2: 6 * p * (1 - p) ** 5, 3: 3 * p ** 2 * (1 - p) ** 4,
                     4: 3 * p ** 2 * (1 - p) ** 4, 5: 6 * p ** 2 * (1 - p) ** 4, 6: 3 * p ** 2 * (1 - p) ** 4,
                     7: 6 * p ** 3 * (1 - p) ** 3, 8: 6 * p ** 3 * (1 - p) ** 3, 9: 3 * p ** 4 * (1 - p) ** 2,
                     10: 6 * p ** 3 * (1 - p) ** 3, 11: 2 * p ** 3 * (1 - p) ** 3, 12: 3 * p ** 4 * (1 - p) ** 2,
                     13: 6 * p ** 4 * (1 - p) ** 2, 14: 3 * p ** 4 * (1 - p) ** 2, 15: 6 * p ** 5 * (1 - p), 16: p ** 6}
    if printing:
        print('sum is: ' + str(
            probabilities[1] + probabilities[2] + probabilities[3] + probabilities[4] + probabilities[5] +
            probabilities[6] + probabilities[7] + probabilities[8] + probabilities[9] + probabilities[10] +
            probabilities[11] + probabilities[12] + probabilities[13] + probabilities[14] + probabilities[15] +
            probabilities[16]))

    return probabilities


def barchart(connections, probabilites, filename, file):
    """This generates charts so you can visualize data
    It generates one barchart for all 16 connections, then 3 smaller ones for the less frequent edge numbers
    It is also where I output connections and probabilities to the file"""

    file.write('connections by type:')
    file.write('\n')
    file.write(str(connections))
    file.write('\n\n')
    file.write('probabilities from math:')
    file.write('\n')
    file.write(str(probabilites))
    file.write('\n\n')

    # graph count of different connection types
    plt.style.use('ggplot')
    x = connections.keys()
    g = connections.values()
    x_pos = np.arange(len(x))
    plt.bar(x_pos, g, color='#7ed6df')
    plt.xlabel("Triple Type")
    plt.ylabel("Count")
    plt.title("Frequecy of 3-way Connections in " + filename)
    plt.xticks(x_pos, x)

    if save_graphs:
        plt.savefig(filename + '_counts.png')
    if show_graphs:
        plt.show()
    plt.close()

    # turn counts into percentages
    total_connections = 0
    for k in connections.keys():
        total_connections += connections[k]

    for k in connections.keys():
        connections[k] = connections[k] / total_connections

    file.write('probabilities from data:')
    file.write('\n')
    file.write(str(connections))
    file.write('\n\n')
    file.close()

    plt.style.use('ggplot')  # graph occuring percentages vs calculated
    width = 0.35
    x = connections.keys()
    g = connections.values()
    f = probabilites.values()
    x_pos = np.arange(len(x))
    plt.bar(x_pos - width / 2, f, width, color='#e532fd')
    plt.bar(x_pos + width / 2, g, width, color='#7ed6df')
    plt.xlabel("Triple Type")
    plt.ylabel("Count")
    plt.title("Frequecy of 3-way Connections in " + filename)
    plt.xticks(x_pos, x)
    if save_graphs:
        plt.savefig(filename + '_frequency.png')
    if show_graphs:
        plt.show()
    plt.close()

    plt.style.use('ggplot')  # graph occuring percentages vs calculated for 2 edge connections
    width = 0.35
    x = [3, 4, 5, 6]
    g = []
    f = []
    for i in x:
        g.append(connections[i])
    for i in x:
        f.append(probabilites[i])
    x_pos = np.arange(len(x))
    plt.bar(x_pos - width / 2, f, width, color='#e532fd')
    plt.bar(x_pos + width / 2, g, width, color='#7ed6df')
    plt.xlabel("Triple Type")
    plt.ylabel("Count")
    plt.title("Frequecy of 3-way Connections with 2 edges " + filename)
    plt.xticks(x_pos, x)

    if save_graphs:
        plt.savefig(filename + '_frequency_2edges.png')
    if show_graphs:
        plt.show()
    plt.close()

    plt.style.use('ggplot')  # graph occuring percentages vs calculated for 3 edge connections
    width = 0.35
    x = [7, 8, 10, 11]
    g = []
    f = []
    for i in x:
        g.append(connections[i])
    for i in x:
        f.append(probabilites[i])
    x_pos = np.arange(len(x))
    plt.bar(x_pos - width / 2, f, width, color='#e532fd')
    plt.bar(x_pos + width / 2, g, width, color='#7ed6df')
    plt.xlabel("Triple Type")
    plt.ylabel("Count")
    plt.title("Frequecy of 3-way Connections with 3 edges 3" + filename)
    plt.xticks(x_pos, x)

    if save_graphs:
        plt.savefig(filename + '_frequency_3edges.png')
    if show_graphs:
        plt.show()
    plt.close()

    plt.style.use('ggplot')  # graph occuring percentages vs calculated for 4, 5, and 6 edge connections
    width = 0.35
    x = [9, 12, 13, 14, 15, 16]
    g = []
    f = []
    for i in x:
        g.append(connections[i])
    for i in x:
        f.append(probabilites[i])
    x_pos = np.arange(len(x))
    plt.bar(x_pos - width / 2, f, width, color='#e532fd')
    plt.bar(x_pos + width / 2, g, width, color='#7ed6df')
    plt.xlabel("Triple Type")
    plt.ylabel("Count")
    plt.title("Frequecy of 3-way Connections with 3 edges 3" + filename)
    plt.xticks(x_pos, x)

    if save_graphs:
        plt.savefig(filename + '_frequency_456edges.png')
    if show_graphs:
        plt.show()
    plt.close()
