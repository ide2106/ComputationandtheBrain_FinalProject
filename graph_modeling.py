# You need to rename these to find the files in your computer if you want to run this code
import OtherCode.ComputationAndTheBrain.graph_generation as graph_generation
import OtherCode.ComputationAndTheBrain.graph_connections as graph_connections
import networkx
import scipy
import numpy
from scipy.optimize import minimize

# 'ConnectomeGraphs/data_rattus.norvegicus_brain_1.graphml.txt'
# 'ConnectomeGraphs/data_mouse_visual.cortex_1.graphml.txt'
# 'ConnectomeGraphs/data_mouse_brain_1.graphml.txt'
GOAL_GRAPH = 'ConnectomeGraphs/data_mixed.species_brain_1.graphml.txt'
OUTPUT = 'mouse_output.txt'
printing = True


def main():
    """This looks at the GOAL_GRAPH variable and gets the connectivity distribution from the appropriate file,
    and attempts to generate an ROC graph that has similar connectivity"""
    n, d, goal_dist = read_goal()
    if printing:
        print(n, d, goal_dist)

    with open(OUTPUT, 'w') as output_file:
        graph = create_model(n, d, goal_dist, output_file)
        networkx.write_graphml(graph, 'model_graph.graphml')


def create_model(n, d, goal_dist, file):
    """Takes graph size n, connectivity d, and a goal connectivity distribution and attempts to generate
    an ROC graph with these parameters by a (sort of) gradient descent over the grid of parameters.
    Note: this function currently only tries to optimize one community type to fit the graph,
    with b=0 or b=s/2"""

    test_size = n
    curr_s = test_size//2
    curr_d = d
    curr_b = 0
    curr_dist = [(curr_s, curr_d, curr_b)]

    # this part was used for me to visualize how things started
    #
    # starting_point = graph_generation.roc_generation(curr_dist, n, d)
    # connections = graph_connections.get_connections(starting_point, file)
    # cat_data = {1: 0.6934838235420362, 2: 0.07870262974956996, 3: 0.0058078895000823125, 4: 0.14106704573000958,
    #            5: 0.0010064858362765648, 6: 0.014418771622477839, 7: 0.009029203835602222, 8: 0.0009853326067878817,
    #            9: 0.028175247961144078, 10: 0.006974257705431414, 11: 8.252605226526663e-06, 12: 0.011328929825394319,
    #            13: 0.00019692423506056724, 14: 0.0011571195960435113, 15: 0.006257514198631006,
    #            16: 0.0014005714502260478}
    #
    # with open('graphing1'+str(OUTPUT), 'w') as output_file1:
    #    graph_connections.barchart(connections, cat_data, 'ROCvsCat_initial', output_file1)

    args = [test_size, d, goal_dist, file]
    curr_e = test_model(curr_dist, args)

    if printing:
        print('STARTING TESTING ERROR: '+str(curr_e))
        print()
    args2 = [n, d, goal_dist, file]
    error = test_model(curr_dist, args2)
    if printing:
        print('STARTING FULL SIZE ERROR: '+str(error))
        print()

    # do magic for better model
    steps = 0
    learning_rate = [test_size//8, 10]
    s_options = [4, 0, -4]
    d_options = [0.05, 0, -0.05]
    b_options = [0, 1]
    while learning_rate[0] * 4 > 0:
        steps += 1
        if steps % 3 == 0:
            learning_rate[0] = learning_rate[0] // 2
            learning_rate[1] = learning_rate[1] / 2
            if printing:
                print(steps, learning_rate, curr_e)
                print('testing')
                print((curr_s, curr_d, curr_b))
                print()
                print()

        best_option = [0, 0, 0]
        for s_change in s_options:
            s_new = curr_s + s_change
            if s_new < 3:
                s_new = 3
            elif s_new > test_size:
                s_new = test_size
            for d_change in d_options:
                d_new = curr_d + d_change
                if d_new < 0.001:
                    d_new = 0.001
                elif d_new > 1:
                    d_new = 1
                for b_change in b_options:
                    b_new = b_change
                    dist_new = [(s_new, d_new, b_new)]
                    e_new = test_model(dist_new, args)
                    if e_new < curr_e:
                        curr_e = e_new
                        best_option = [s_change, d_change, b_change]
        curr_s = curr_s + best_option[0]*learning_rate[0]
        if curr_s > test_size:
            curr_s = test_size
        elif curr_s < 3:
            curr_s = 3
        curr_d = curr_d + best_option[1]*learning_rate[1]
        if curr_d < 0.001:
            curr_d = 0.001
        elif curr_d > 1:
            curr_d = 1
        curr_b = best_option[2]

    if printing:
        print()
        print('calculating end')
    dist = [(curr_s, curr_d, curr_b)]
    args = [test_size, d, goal_dist, file]
    error = test_model(dist, args)
    if printing:
        print('FINAL TEST SIZE ERROR: '+str(error))
        print()
    args2 = [n, d, goal_dist, file]
    error_final = test_model(dist, args2)
    if printing:
        print('FINAL FULL SIZE ERROR: '+str(error_final))
        print()

    ending_point = graph_generation.roc_generation(dist, n, d)
    return ending_point


def test_model(dist, args):
    """Takes a distribution and graph parameters, and tests them by generating and ROC graph with the parameters,
    Then returning the error between that and the goal distribution"""

    n, d, goal_dist, file = args
    graph = graph_generation.roc_generation(dist, n, d)
    connections = graph_connections.get_connections(graph, file)
    total_connections = 0
    for k in connections.keys():
        total_connections += connections[k]
    for k in connections.keys():
        connections[k] = connections[k]/total_connections
    error = calculate_error(connections, goal_dist)
    return error


def calculate_error(curr_dist, goal_dist):
    count = 0
    error = 0
    for k in curr_dist.keys():
        count += 1
        error += (curr_dist[k] - goal_dist[k])**2
    return error


def read_goal():
    if printing:
        print('reading')
    input_file = open(GOAL_GRAPH, 'r')
    flag = 0
    for line in input_file:
        if 'nodes' in line:
            n = int(line[:-7])
        elif 'p is' in line:
            p = float(line[5:])
        elif 'probabilities from data:' in line:
            flag = 1
        elif flag == 1:
            goal_probabilites = {}
            line = line.strip('{').strip('}\n')
            tokens = line.split(',')
            for t in tokens:
                key, value = t.split(':')
                goal_probabilites[int(key)] = float(value)
            flag = 0

    return n, p, goal_probabilites


main()
