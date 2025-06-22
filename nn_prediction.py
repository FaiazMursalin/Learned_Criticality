import numpy as np
import pandas as pd
import cv2
import glob
import os
import tensorflow as tf
from keras.models import Sequential
# from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt


def extract_filename(fp):
    return os.path.basename(fp).split(".")[0]
    
def extract_local_features(image, x, y, size=28):
    half_size = size // 2
    max_y, max_x = image.shape[0], image.shape[1]
    x = max(half_size, min(max_x - half_size, x))
    y = max(half_size, min(max_y - half_size, y))
    return image[y-half_size:y+half_size, x-half_size:x+half_size]


def read_environment(file_path):
    im = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    im = cv2.flip(im, 0)
    im = im / 255.0
    im = np.pad(im, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    return im

def create_test_features(image_path, samples_path):

    features = {}
    image = read_environment(image_path)
    dataframe = pd.read_csv(samples_path,  names=["x", "y", "centrality"])  # Load the DataFrame if it's a path

    for i, row in dataframe.iterrows():
        x, y, centrality = int(row['x']), int(row['y']), row['centrality']
        cropped_image = extract_local_features(image, x, y, size=28)
        features[(x,y)] = cropped_image.reshape(-1, 28, 28, 1)
    return features



input_image = f"Images/predictions/library_resized.png"
img = read_environment(input_image)
filename = extract_filename(input_image)
ttv = input_image.split(os.path.sep)[1]
bc_in_file = os.path.join("BC", ttv, f"{filename}_bc.txt")

test_features = create_test_features(input_image, bc_in_file)


model = load_model("best_model.h5")
samples = list(test_features.keys())
test_input = np.squeeze(np.array(list(test_features.values())), axis=1)
predictions = model.predict(test_input)

# sorted_indices = np.argsort(predictions, axis=0)[::-1]

# # Get the indices of the top 10 predictions
# top_indices = sorted_indices[:10].flatten()

# # Plot the top 10 predictions along with their corresponding images
# height, width = img.shape

# # Initialize lists to store x, y coordinates
# x_coords = []
# y_coords = []

# # Iterate through the array to identify free space coordinates
# for y in range(height):
#     for x in range(width):
#         if img[y, x] == 0:  # Assuming 1 represents free space
#             x_coords.append(x)
#             y_coords.append(y)

# plt.figure(figsize=(10, 5))
# plt.plot(x_coords, y_coords, ".k")
# plt.grid(True)
# plt.axis("equal")
# plt.gca().invert_yaxis()
# for i, idx in enumerate(top_indices):
#     plt.subplot(2, 5, i + 1)
#     # plt.imshow(samples[idx].reshape(28, 28), cmap='gray')
#     x, y = samples[idx]
#     plt.plot([x], [y],".r")
#     print(x, y)
#     # plt.plot(samples[idx][0])
#     plt.text(x, y, f'{predictions[idx][0]:.10f}', color='red', fontsize=8)
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

sorted_indices = np.argsort(predictions, axis=0)[::-1]

# Get the indices of the top 100 predictions
top_indices = sorted_indices[:2].flatten()

# Plot the top 100 predictions along with their corresponding (x, y) coordinates
plt.figure(figsize=(10, 10))
for idx in top_indices:
    
    # Plot the original image
    plt.imshow(img, cmap='gray')
    
    # Get the (x, y) coordinate for the current prediction
    x, y = samples[idx]
    
    # Plot the (x, y) coordinate using plot
    plt.plot(x, y, marker='o', markersize=5, color='red')
    
    # Add prediction value as text
    # plt.text(x, y, f'{predictions[idx][0]:.10f}', color='red', fontsize=8)  
    
plt.title('Top 100 Predictions with (x, y) Coordinates')
# plt.axis('off')
# plt.grid()
plt.show()