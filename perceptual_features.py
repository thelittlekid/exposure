import numpy as np
import tensorflow as tf
import collections
from PIL import Image
# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
# Most code in this file was borrowed from https://github.com/anishathalye/neural-style/blob/master/vgg.py

import scipy.io
import simple_cache

CONTENT_LAYERS_ITEMS = ['relu4_2']
CONTENT_LAYER_WEIGHTS = [1.0]
STYLE_LAYERS_ITEMS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
STYLE_LAYER_WEIGHTS = [.2, .2, .2, .2, .2]

STYLE_LAYERS = {}
for layer, weight in zip(STYLE_LAYERS_ITEMS, STYLE_LAYER_WEIGHTS):
    STYLE_LAYERS[layer] = weight

CONTENT_LAYERS = {}
for layer, weight in zip(CONTENT_LAYERS_ITEMS, CONTENT_LAYER_WEIGHTS):
    CONTENT_LAYERS[layer] = weight

CONTENT_LAYERS = collections.OrderedDict(sorted(CONTENT_LAYERS.items()))
STYLE_LAYERS = collections.OrderedDict(sorted(STYLE_LAYERS.items()))

# download URL : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
#MODEL_FILE_NAME = 'imagenet-vgg-verydeep-19.mat'
def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)

def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

def preprocess(image, mean_pixel):
    return image - mean_pixel

def undo_preprocess(image, mean_pixel):
    return image + mean_pixel

class VGG19:
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    def __init__(self, model_path):
        data = scipy.io.loadmat(model_path)

        self.mean_pixel = np.array([123.68, 116.779, 103.939])

        self.weights = data['layers'][0]


    def preprocess(self, image):
        return image-self.mean_pixel

    def undo_preprocess(self,image):
        return image+self.mean_pixel

    def setup_vgg19_net(self, img, scope='vgg19'):
        net = {}
        current = img

        with tf.variable_scope(scope):
            for i, name in enumerate(self.layers):
                kind = name[:4]
                if kind == 'conv':
                    kernels = self.weights[i][0][0][2][0][0]
                    bias = self.weights[i][0][0][2][0][1]

                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = np.transpose(kernels, (1, 0, 2, 3))
                    bias = bias.reshape(-1)

                    current = _conv_layer(current, kernels, bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current)
                elif kind == 'pool':
                    current = _pool_layer(current)
                net[name] = current

        assert len(net) == len(self.layers)
        return net


def _gram_matrix( tensor, shape=None):

    if shape is not None:
        B = shape[0]  # batch size
        HW = shape[1] # height x width
        C = shape[2]  # channels
        CHW = C*HW
    else:
        B, H, W, C = map(lambda i: i.value, tensor.get_shape())
        HW = H*W
        CHW = W*H*C

    # reshape the tensor so it is a (B, 2-dim) matrix
    # so that 'B'th gram matrix can be computed
    feats = tf.reshape(tensor, (B, HW, C))

    # leave dimension of batch as it is
    feats_T = tf.transpose(feats, perm=[0, 2, 1])

    # paper suggests to normalize gram matrix by its number of elements
    gram = tf.matmul(feats_T, feats) / CHW

    return gram


class PerceptualFeatures(object):

    def __init__(self, sess, vgg19_model_path):

        self.sess = sess

        self.vgg_net = VGG19(vgg19_model_path)

        self.width = 224
        self.height = 224

        self.img1 = tf.placeholder(tf.float32, shape=(1, 224, 224, 3), name='img')
        self.SetupVGG()
        self.SetupTensorflow()

    def SetupVGG(self):
        self.feed_forward_net1 = self.vgg_net.setup_vgg19_net(self.img1)


    # Right now I am using the first layer of the STYLE_LAYERS, this should be changed for experiments, different layers,
    # or using a gram matrix and so on ...
    def SetupTensorflow(self):
        self.F = self.feed_forward_net1[STYLE_LAYERS_ITEMS[4]]

    # This creates also a cache file which stores the entries for one hour, delete the cache file to prune.
    def GetFeature(self, file_path):
        cache_value = (simple_cache.load_key("perceptual.cache",file_path))
        if cache_value is None:
            image1 = np.asarray(Image.open(file_path).resize((224, 224)))
            image_1_pre = self.vgg_net.preprocess(np.asarray(image1)).reshape(1, self.width, self.height, 3)

            feed_dict = {self.img1: image_1_pre}

            feature = self.sess.run(self.F, feed_dict=feed_dict).flatten()
            print(feature.shape)
            simple_cache.save_key("perceptual.cache",file_path,feature,3600)
            return feature
        else:
            print("loading from cache for %s"%file_path)
            return cache_value

