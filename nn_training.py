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
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt

 
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, image_paths, dataframe_paths, batch_size=32, n_channels=3, shuffle=True):
        'Initialization'
        # self.dim = dim
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.dataframe_paths = dataframe_paths
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
 
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_paths) / self.batch_size))
 
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_images = [self.image_paths[k] for k in indexes]
        list_dataframes = [self.dataframe_paths[k] for k in indexes]
        X, y = self.__data_generation(list_images, list_dataframes)
        return X, y
 
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
 
    def __data_generation(self, list_images, list_dataframes):
        'Generates data containing batch_size samples'
        features = []
        labels = []
 
        for image_path, dataframe_path in zip(list_images, list_dataframes):
            image = self.read_environment(image_path)
 
            dataframe = pd.read_csv(dataframe_path,  names=["x", "y", "centrality"])  # Load the DataFrame if it's a path
            # If dataframe is already an object, remove pd.read_csv and use dataframe_path directly
 
            for i, row in dataframe.iterrows():
                x, y, centrality = int(row['x']), int(row['y']), row['centrality']
                cropped_image = self.extract_local_features(image, x, y, size=28)
                features.append(cropped_image)
                labels.append(centrality)
 
        return np.array(features), np.array(labels)
 
    @staticmethod
    def extract_local_features(image, x, y, size=28):
        half_size = size // 2
        max_y, max_x = image.shape[0], image.shape[1]
        x = max(half_size, min(max_x - half_size, x))
        y = max(half_size, min(max_y - half_size, y))
        return image[y-half_size:y+half_size, x-half_size:x+half_size]
    
    @staticmethod
    def read_environment(file_path):
        im = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        im = cv2.flip(im, 0)
        im = im / 255.0
        im = np.pad(im, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        return im



def extract_filename(fp):
    return os.path.basename(fp).split(".")[0]




def plot_history(history, savefilename):
    plt.figure(figsize=(12,6), dpi=300.0)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    if 'mae' in history.history:
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Performance')
    plt.ylabel('Loss / MAE')
    plt.xlabel('Epoch')
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.grid()
    plt.savefig(savefilename)



def create_model():
    # Build the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=["mae"])

    return model



fp_s = glob.glob("Images/**/*.png")

data = {"test": {}, "train": {}, "validation": {}}

for fp in fp_s:
    filename = extract_filename(fp)
    ttv = fp.split(os.path.sep)[1]
    bc_in_file = os.path.join("BC", ttv, f"{filename}_bc.txt")

    data[ttv][fp] = bc_in_file

train_generator = DataGenerator(list(data["train"].keys()), list(data["train"].values()), batch_size=32)
validation_generator = DataGenerator(list(data["validation"].keys()), list(data["validation"].values()), batch_size=32)

model = create_model()

# Callbacks
checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', mode='min')
early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss', mode='min')
 
# Fit the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=300,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

plot_history(history, "plots.png")

model.save("cprm_model.h5")