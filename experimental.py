import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from net import GAN
from util import load_config
import shutil
import numpy as np
import glob
import cv2
import util
import ffmpeg
import scipy
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from PIL import Image
import tensorflow as tf
import random

FILE_PATH = 'data/artists/FiveK_C/0001.jpg'
FiveK_C_FOLDER = '/Users/yifan/Documents/GitHub/exposure/data/artists/FiveK_C/'
RETRAIN_DATA_FOLDER = '/Users/yifan/Dropbox/stylish photos/Data/FiveK_C_retrain/'


def load_pretrained_networks(config_name, model_dir, model_name, iternum=20000):
    """
    Load pretrained networks from the exposure repository
    :param model_dir: root directory that contains the model folders
    :param config_name: suffix in the name of the configuration file
    :param model_name: name of the model
    :param iternum: number of max iteration
    :return: the instance of GAN class that contains the pretrained networks
    """
    print("Loading pretrained networks...")
    shutil.copy('%s/%s/%s/scripts/config_%s.py' %
                (model_dir, config_name, model_name, config_name), 'config_tmp.py')
    cfg = load_config('tmp')
    cfg.name = config_name + '/' + model_name
    net = GAN(cfg, restore=True)
    net.restore(iternum)
    return net


def load_images(img_paths):
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)[:, :, ::-1]
        imgs.append(np.asarray(cv2.resize(img, dsize=(64, 64))))

        # imgs.append(np.asarray(Image.open(img_path).resize((64, 64))))
    return np.stack(imgs, axis=0) / 255.0


def get_discriminator_logits(net, imgs, batch_size=64):
    D_logits = []

    # Compute the logits batch by batch
    for starti in range(0, len(imgs), batch_size):
        endi = min(starti + batch_size, len(imgs))
        img_batch = imgs[starti:endi, :, :, :]
        logit_batch = net.sess.run(net.real_logit, feed_dict={net.real_data: img_batch})
        D_logits.append(logit_batch.flatten())

    return np.hstack(D_logits)

def compute_emd(net, imgs_real, imgs_fake):
    return net.sess.run(net.emd, feed_dict={net.real_data: imgs_real, net.fake_output: imgs_fake})


def test_discriminator_logits():
    # img_dir = sys.argv[3]
    img_dir_real = './data/artists/Ansel_Adams'
    img_dir_fake = './data/artists/koda'

    # Get the list of image paths in the directory
    img_paths_real = glob.glob(img_dir_real + '**/*.jpg', recursive=True)
    img_paths_fake = glob.glob(img_dir_fake + '**/*.jpg', recursive=True)
    img_num = min(len(img_paths_real), len(img_paths_fake))
    img_paths_real, img_paths_fake = img_paths_real[0:img_num], img_paths_fake[0:img_num]
    print("Number of images in the directory: ", img_num)

    # Load the pretrained network
    net = load_pretrained_networks(config_name='example', model_dir='models', model_name='Ansel_Adams')

    # Load the images
    imgs_real = load_images(img_paths_real)
    imgs_fake = load_images(img_paths_fake)
    # imgs_fake = np.random.random(imgs_real.shape)
    # imgs_fake = np.power(imgs_fake, 2.2)

    # Compute logits using the discriminator
    D_logits_real = get_discriminator_logits(net, imgs_real)
    D_logits_fake = get_discriminator_logits(net, imgs_fake)

    # Compute earth mover's distance
    emd = compute_emd(net, imgs_real, imgs_fake)
    print("Earth mover's distance (real - fake): ", emd)
    pass

    """
    The discriminator always outputs a positive number, even for zeros, ones, and random.rand. The output is no longer a 
    probability. The earth mover's distance is quite close for jpeg datasets. 
    """


def reorder_pixels_by_intensity():
    """
    Reordering pixels according to the intensity
    """

    img = cv2.imread(FILE_PATH)[:, :, ::-1]
    img_1d = img.reshape((-1, 3))
    intensity = np.sum(img, axis=2).ravel()
    argseq = np.argsort(intensity)

    img_1d_ = img_1d[argseq, :]
    img_ = img_1d_.reshape(img.shape)

    util.display_image(img_)


if __name__ == "__main__":


    pass





