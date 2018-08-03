from PIL import Image
import cv2
import numpy as np
from numpy import random
import pickle
from sklearn.cluster import KMeans
from glob import glob
import os
from shutil import copy2
import util


FILE_PATH = 'data/artists/FiveK_C/0001.jpg'
FILE_PATH = 'data/artists/Ansel_Adams/Afternoon sun.jpg'
FiveK_C_FOLDER = '/Users/yifan/Documents/GitHub/exposure/data/artists/FiveK_C/'
# FiveK_C_FOLDER = '/Users/yifan/Dropbox/stylish photos/Data/FiveK_C_VsYCbCr_gray/'
RETRAIN_DATA_FOLDER = '/Users/yifan/Dropbox/stylish photos/Data/FiveK_C_8clustered_VsGrayColor/'
OUTPUT_FOLDER = '/Users/yifan/Dropbox/stylish photos/Data/FiveK_C_reordered_2/'

# image = cv2.imread(FILE_PATH)[:, :, ::-1]  # read image in r-g-b order
#
# img = Image.fromarray(image, 'RGB')
# img.show()

# data = np.load('train_image.npy')
#
# image = np.uint8(data[0, :, :, :] * 255)
# img = Image.fromarray(image, 'RGB')
# img.show()


def cluster_with_haystack():
    """
    Clustering the images based on the haystack features.
    Note: the Haystack feature is computed outside of this repository using the imagecore
    :return: void
    """
    # Extract features using pretrained networks
    # Clustering images based on features extracted by pretrained networks
    # Load pre-computed features in cache
    cache_path = '/Users/yifan/Dropbox/stylish photos/inputs/5k_feature_vectorscope_gray_haystack.pkl'
    with open(cache_path, 'rb') as f:
        features = pickle.load(f)

    haystack_features = features[0]['features']
    kmeans = KMeans(n_clusters=8, random_state=0).fit(haystack_features)

    files = glob(FiveK_C_FOLDER + '*.jpg', recursive=False)
    files.sort()

    labels = kmeans.labels_

    # Create subfolders for each cluster
    for i in range(max(labels)+1):
        subfolder = RETRAIN_DATA_FOLDER + str(i) + '/'
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    # Copy images to each cluster
    for i in range(len(files)):
        file, label = files[i], labels[i]
        subfolder = RETRAIN_DATA_FOLDER + str(label) + '/'
        copy2(file, subfolder)


def shuffle_image_pixels():
    """
    Reorder pixels with various block size for all images, then store them
    :return:
    """
    files = glob(FiveK_C_FOLDER + '*.jpg', recursive=False)
    files.sort()

    for block in [1, 2, 4, 8, 16, 32, 64]:
        OUTPUT_FOLDER = '/Users/yifan/Dropbox/stylish photos/Data/FiveK_C_reordered_' + str(block) + '/'
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        for i in range(len(files)):
            file = files[i]
            img = cv2.imread(file)[:, :, ::-1]
            print(img.shape)
            img_ = util.reorder_pixels(img, [block, block])
            output_path = OUTPUT_FOLDER + os.path.basename(file)
            cv2.imwrite(output_path, img_[:, :, ::-1])

            # if i == 10:
            #     break


def cluster_with_multi_haystack():
    cache_path = '/Users/yifan/Dropbox/stylish photos/inputs/5k_feature_vectorscope_gray_haystack.pkl'
    haystack_features_1 = load_features(cache_path)

    cache_path = '/Users/yifan/Dropbox/stylish photos/inputs/5k_feature_vectorscope_color_haystack.pkl'
    haystack_features_2 = load_features(cache_path)

    haystack_features = []
    for i in range(len(haystack_features_1)):
        combined_feature = np.concatenate((haystack_features_1[i], haystack_features_2[i]))
        haystack_features.append(combined_feature)

    kmeans = KMeans(n_clusters=8, random_state=0).fit(haystack_features)

    files = glob(FiveK_C_FOLDER + '*.jpg', recursive=False)
    files.sort()

    labels = kmeans.labels_

    # Create subfolders for each cluster
    for i in range(max(labels) + 1):
        subfolder = RETRAIN_DATA_FOLDER + str(i) + '/'
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    # Copy images to each cluster
    for i in range(len(files)):
        file, label = files[i], labels[i]
        subfolder = RETRAIN_DATA_FOLDER + str(label) + '/'
        copy2(file, subfolder)


def load_features(cache_path):
    """
    Load pre-computed features from .pkl file
    :param cache_path:
    :return: a list of numpy arrays, each element in the list corresponds to the feature for an input image
    """
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    return cache[0]['features']


if __name__ == "__main__":
    cluster_with_multi_haystack()
    pass

