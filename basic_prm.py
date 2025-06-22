import glob
import os

import numpy as np
import networkx as nx
import cv2
import matplotlib.pyplot as plt
import tqdm


# Read the environment image
def read_environment(file_path):
    im = cv2.imread(file_path)
    im = cv2.flip(im, 0)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im / 255.0
    im = np.pad(im, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    return im


def read_coordinates(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split(','))
            coordinates.append((x, y))
    return coordinates
# Convert image to binary occupancy grid

def calculate_betweenness_centrality(G):
    betweenness_centrality = nx.betweenness_centrality(G)
    return betweenness_centrality

# Check if there is a clear path between two points
def is_clear_path(img_binary, p1, p2):
    # Bresenham's line algorithm for line-of-sight check
    x0, y0 = int(p1[0]), int(p1[1])
    x1, y1 = int(p2[0]), int(p2[1])

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        if x0 < 0 or y0 < 0 or x0 >= img_binary.shape[1] or y0 >= img_binary.shape[0] or img_binary[y0, x0] == 0:
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return True

# Construct the graph by connecting nearby coordinates
def construct_graph(coordinates, img_binary, threshold_distance):
    G = nx.Graph()
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            if np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j])) <= threshold_distance:
                if is_clear_path(img_binary, coordinates[i], coordinates[j]):
                    G.add_edge(coordinates[i], coordinates[j])
    return G

# # Visualize the graph
# def visualize_graph(G, img):
#     # plt.imshow(img, cmap='gray')
#     # plt.imshow(cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), -1), cmap='gray')
#     pos = {node: (node[0], img.shape[1] - node[1]) for node in G.nodes()}  # Invert y-axis for plotting
#     # pos = {node: (node[0], img.shape[0] - node[1]) for node in G.nodes()}  # No y-axis inversion
#     nx.draw(G, pos, with_labels=False, node_size=2, edge_color='b', alpha=0.5)
#     plt.title('PRM Graph in Environment')
#     plt.show()

# Example usage




if __name__ == "__main__":

    samples_basepath = "Samples_dataset/"
    images_basepath = "Images/predictions/*.png"
    filepaths = glob.glob(images_basepath)

    for fp in tqdm.tqdm(filepaths):
        ttv = fp.split(os.path.sep)[1]
        filename = os.path.basename(fp).split(".")[0]
        samples_filename = os.path.join(samples_basepath, ttv, f"{filename}_samples.txt")
        bc_out_file = os.path.join("BC", ttv, f"{filename}_bc.txt")

        # Read the environment image
        img = read_environment(fp)
        # print(img)
        # Sample coordinates (replace this with your PRM sampling results)
        coordinates = read_coordinates(samples_filename)

        # Define the distance threshold for connecting nearby coordinates
        threshold_distance = 100

        # Construct the graph by connecting nearby coordinates
        G = construct_graph(coordinates, img, threshold_distance)
        # print(f"started calculating betweenness centrality for {filename}")
        betweenness_centrality = calculate_betweenness_centrality(G)
        with open(bc_out_file, "w") as file:
            for node, centrality in betweenness_centrality.items():
                file.write(f"{node[0]},{node[1]},{centrality}\n")


        raise SystemExit