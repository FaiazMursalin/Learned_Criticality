import glob
import os
import numpy as np
import networkx as nx
import cv2
import tqdm
import random
import re
from concurrent.futures import ProcessPoolExecutor

samples_basepath = "Samples_dataset/"
images_basepath = "Images/**/*.png"
filepaths = glob.glob(images_basepath)



# Number of image to select from each split
N_TRAIN = 4400
N_TEST = 500
N_VALIDATION = 500

# test and validation images change every 101
# train images change every 801
TRAIN_PERIODICITY = 801
TEST_PERIODICITY = 101
VALIDATION_PERIODICITY = 101


def extract_filename(fp):
    return os.path.basename(fp).split(".")[0]


def select_random_images(image_paths, k_train, k_test, k_validation, train_periodicity, test_periodicity, validation_periodicity):
    # Sort the image paths numerically
    def numerical_sort(value):
        parts = re.findall(r'(\d+)', value)
        return tuple(int(part) if part.isdigit() else part for part in parts)
    
    image_paths.sort(key=numerical_sort)

    # Separate image paths into train, test, and validation directories based on periodicity
    train_images = [path for path in image_paths if "train" in path]
    test_images = [path for path in image_paths if "test" in path]
    validation_images = [path for path in image_paths if "validation" in path]

    # Select images from each directory while maintaining distribution
    selected_train_images = select_images_with_periodicity(train_images, k_train, train_periodicity)
    selected_test_images = select_images_with_periodicity(test_images, k_test, test_periodicity)
    selected_validation_images = select_images_with_periodicity(validation_images, k_validation, validation_periodicity)

    # Combine selected images from all directories into a single list
    selected_images = selected_train_images + selected_test_images + selected_validation_images

    return selected_images

def select_images_with_periodicity(image_paths, k, periodicity):
    # Ensure that at least one image is selected from each period
    selected_images = [image_paths[i] for i in range(0, len(image_paths), periodicity)]

    # Select the remaining random images from the remaining pool
    remaining_images = [image for image in image_paths if image not in selected_images]
    selected_images += random.sample(remaining_images, k - len(selected_images))

    return selected_images


def read_environment(file_path):
    im = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    im = cv2.flip(im, 0)
    im = im / 255.0
    im = np.pad(im, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    return im

def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        coordinates = [tuple(map(float, line.strip().split(','))) for line in file]
    return coordinates

def is_clear_path(img_binary, p1, p2):
    x0, y0 = int(p1[0]), int(p1[1])
    x1, y1 = int(p2[0]), int(p2[1])
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
    err = dx - dy
    while x0 != x1 or y0 != y1:
        if img_binary[y0, x0] == 0:
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return True

def construct_graph(args):
    coordinates, img_binary, threshold_distance = args
    G = nx.Graph()
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            if np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j])) <= threshold_distance:
                if is_clear_path(img_binary, coordinates[i], coordinates[j]):
                    G.add_edge(coordinates[i], coordinates[j])
    return G

def process_image(fp):
    filename = os.path.basename(fp).split(".")[0]
    ttv = fp.split(os.path.sep)[1]
    samples_filename = os.path.join(samples_basepath, ttv, f"{filename}_samples.txt")
    bc_out_file = os.path.join("BC", ttv, f"{filename}_bc.txt")

    if os.path.exists(bc_out_file):
        print(f"{bc_out_file} exists.")
        return

    img = read_environment(fp)
    coordinates = read_coordinates(samples_filename)
    threshold_distance = 100

    G = construct_graph((coordinates, img, threshold_distance))
    betweenness_centrality = nx.betweenness_centrality(G)
    
    with open(bc_out_file, "w") as file:
        for node, centrality in betweenness_centrality.items():
            file.write(f"{node[0]},{node[1]},{centrality}\n")

if __name__ == "__main__":

    already_done_filename = set([extract_filename(fp)[:-3] for fp in glob.glob("BC/**/*")])
    filtered = [fp for fp in (set([extract_filename(f) for f in filepaths])-  already_done_filename)]

    filtered_filepaths = [fp for fp in filepaths if extract_filename(fp) in filtered]

    selected_filepaths = select_random_images(filtered_filepaths, N_TRAIN, N_TEST, N_VALIDATION, TRAIN_PERIODICITY, TEST_PERIODICITY, VALIDATION_PERIODICITY)
    print(f"Selected {len(selected_filepaths)} filepaths to process..")
    with ProcessPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(process_image, selected_filepaths), total=len(selected_filepaths), desc="Processing.."))
