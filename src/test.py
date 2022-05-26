from node import *
import random
import matplotlib.pyplot as plt
import networkx as nx

#ToDo: Test p2p overlay network
#ToDo: Test peer ranking and deranking

MAX_PEERS = 3
LIFETIME = 4
OPPOSIT_FRAC = 0.3
NODES = 20

adj_list = [Node(i,None,None,MAX_PEERS) for i in range(NODES)]

for node in adj_list:
    while len(node.neighbors) < MAX_PEERS:
        node.add_neighbors(random.sample(adj_list,MAX_PEERS))
        if node in node.neighbors:
            node.neighbors = []

#Draw initial graph
graph = build_graph(adj_list,nx.DiGraph())
draw_graph(graph,"G0.png",color_graph(adj_list))

life = 1
for node in adj_list:
    print(node.describe())
while life <= LIFETIME:
    print("###### TRAIN STEP #########")
    #Train every node
    for node in adj_list:
        node.next_candidates(sort="id")
    for node in adj_list:
        node.next_peers(non_iid_frac=OPPOSIT_FRAC)
    graph = build_graph(adj_list,nx.DiGraph())
    draw_graph(graph,"G" + str(life) + ".png",color_graph(adj_list))
    # for node in graph:
    #     print(node.describe())
    #Update every nodes grads
    life += 1
    