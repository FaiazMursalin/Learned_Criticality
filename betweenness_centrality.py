import os

import networkx as nx
import tqdm
import glob

from matplotlib import pyplot as plt


# Read coordinates from the text file
def read_coordinates(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split(','))
            coordinates.append((x, y))
    return coordinates

# Construct a network/graph based on the coordinates
def construct_graph(coordinates):
    G = nx.Graph()
    G.add_nodes_from(coordinates)
    return G

# Calculate betweenness centrality
def calculate_betweenness_centrality(G):
    betweenness_centrality = nx.betweenness_centrality(G)
    return betweenness_centrality

# # Example usage
# file_path = 'coordinates.txt'
# coordinates = read_coordinates(file_path)
# G = construct_graph(coordinates)
# betweenness_centrality = calculate_betweenness_centrality(G)
# print("Betweenness centrality for each node:")
# for node, centrality in betweenness_centrality.items():
#     print(f"Node {node}: {centrality}")

def main():
    basepath = "Samples_dataset/**/*_samples.txt"
    filepaths = glob.glob(basepath)
    # filepaths = ["coordinates.txt"]

    print(list(filepaths))
    for fp in tqdm.tqdm(filepaths):
        print(fp)
        # ttv = fp.split(os.path.sep)[1]
        filename = os.path.basename(fp).split(".")[0]

        coordinates = read_coordinates(fp)
        graph = construct_graph(coordinates)
        betweenness_centrality = calculate_betweenness_centrality(graph)

        pos = nx.spring_layout(graph)  # positions for all nodes
        nx.draw(graph, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=8)
        plt.title('NetworkX Graph from Coordinates')
        plt.show()

        for node, centrality in betweenness_centrality.items():
            if not centrality == 0:
                print(f"Node {node}: {centrality}")

        raise SystemExit


main()