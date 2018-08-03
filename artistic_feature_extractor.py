import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import util

FILE_PATH = 'data/artists/FiveK_C/0001.jpg'
FILE_PATH = '/Users/yifan/Dropbox/stylish photos/100 David Keochkerian/16109638860_f9bae113dc_b.jpg'
FiveK_C_FOLDER = '/Users/yifan/Dropbox/stylish photos/Data/FiveK_C/'
INPUT_FOLDER = '/Users/yifan/Documents/GitHub/non-destructive-style-transfer/data/Unadjusted/'
OUTPUT_FOLDER_A = '/Users/yifan/Documents/GitHub/non-destructive-style-transfer/vectorscopes/color/Unadjusted/'
OUTPUT_FOLDER_B = '/Users/yifan/Documents/GitHub/non-destructive-style-transfer/vectorscopes/gray/Unadjusted/'


def compute_zone_system_features(img, zone_num=11, mode='mask'):
    """
    Compute additional features on color distribution using the "Zone System" proposed by Ansel Adams and Fred Archer
    Each feature map contains pixels of a particular color in a particular zone. The feature map has the same height and
    width as the original input image.

    :param img: the original input image (h x w x d) or the tensor holding the image (? x h x w x d)
    :param zone_num: total number of zones, 11 by default
    :param mode: 'mask' - using binary mask by default; 'color' (or else) - use color map
    :return: zone-system features - numpy array with shape h x w x (zone_num)
            or a tensor with shape (? x h x w x zone_num)
    """
    zone_system_features = []

    # Compute thresholds in the zone system
    thresholds = (1 / zone_num) * np.array(range(zone_num + 1))
    # Handle the boundary thresholds
    thresholds[0] -= 1
    thresholds[-1] += 1

    # Compute zone system features for each zone (outer loop) and each color channel (inner loop)
    # TODO: you may want to switch the order of the loops
    for i in range(len(thresholds) - 1):
        min_value, max_value = thresholds[i], thresholds[i + 1]

        for cidx in range(img.shape[-1]):
            channel = img[..., cidx]

            # Obtain the binary mask of components in the channel that fall into the zone
            if type(img) == tf.Tensor:
                feature_mask = tf.cast(tf.logical_and(min_value <= channel, channel < max_value), dtype=tf.float32)
            else:
                feature_mask = np.logical_and(min_value <= channel, channel < max_value)

            # Compute and append additional features
            if mode == 'mask':
                # Add the binary mask as an additional feature
                if type(img) == tf.Tensor:
                    zone_system_features.append(feature_mask)
                else:
                    zone_system_features.append(feature_mask.astype(channel.dtype))
            else:
                # Add the masked color channel as an additional feature
                if type(img) == tf.Tensor:
                    # TODO: may switch to boolean_mask if it can keep the original dimension in the later version
                    feature_map = tf.multiply(channel, feature_mask)
                else:
                    feature_map = np.copy(channel)  # all numpy arrays are pass by reference
                    feature_map[np.logical_not(feature_mask)] = 0  # masked color channel

                zone_system_features.append(feature_map)

    if type(img) == tf.Tensor:
        return tf.stack(zone_system_features, axis=-1)
    else:
        return np.stack(zone_system_features, axis=-1)


def compute_vectorscope_features_tf(img):
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    Y = (0.299 * R) + (0.587 * G) + (0.114 * B)
    Cb = (-0.169 * R) - (0.331 * G) + (0.499 * B) + 128
    Cr = (0.499 * R) - (0.418 * G) - (0.0813 * B) + 128

    # traditional vectorscope orientation:
    Cr = 256 - Cr


    pass


def compute_vectorscope_features(imgs, percentile=85):
    """
    Compute the vectorscope of images
    :param imgs: numpy array containing images, size h x w x 3
    :return: numpy array containing vectorscope features for the input images, size h x w x d, (d depends on # of planes)
    """
    vectorscopes = []
    for img in imgs:
        vectorscopes.append(compute_vectorscope_base(img, percentile))

    return np.stack(vectorscopes, axis=0)


def compute_vectorscope_base(img, percentile=85, mode='gray', dsize=(64, 64)):
    """
    Compute the vectorscope of a given image
    :param img: the input image in numpy array, h x w x 3
    :param percentile: the percentile used for clipping the gray vectorscope. All values above this will be 256
    :param mode: the mode of vectorscope: gray (1 channel), color (3 channels), and both (4 channels)
    :param dsize: size of the feature map, default (64, 64), should be consistent with the training images
    :return: concatenated vectorscope features in numpy array, h x w x d

    Reference: https://gist.github.com/celoyd/16d075620273e35c71c3d41a524d4cf1
    """
    if img.dtype == np.uint16:
        src = (img / 2 ** 8).astype(np.uint8)

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    Y = (0.299 * R) + (0.587 * G) + (0.114 * B)
    Cb = (-0.169 * R) - (0.331 * G) + (0.499 * B) + 128
    Cr = (0.499 * R) - (0.418 * G) - (0.0813 * B) + 128

    # traditional vectorscope orientation:
    Cr = 256 - Cr

    flat_color = np.zeros((256, 256, 3), dtype=img.dtype)
    gray = np.zeros((256, 256))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            flat_color[int(Cr[x, y]), int(Cb[x, y])] = np.array([R[x, y], G[x, y], B[x, y]])
            gray[int(Cr[x, y]), int(Cb[x, y])] += 1  # build 2d histogram

    # normalize the gray-mode vectorscope
    base_value = np.percentile(gray[np.nonzero(gray)], percentile)
    gray = (gray / base_value * 255).astype('uint8')

    # Resize the feature plane to match the size in image pool
    if not dsize == (256, 256):
        gray = cv2.resize(gray, dsize)
        flat_color = cv2.resize(flat_color, dsize)

    if mode == 'gray':
        return gray[..., np.newaxis]
    elif mode == 'flat_color':
        return flat_color
    elif mode == 'separate':
        return flat_color, gray
    else:
        return np.append(flat_color, gray[..., np.newaxis], axis=-1)


def output_vectorscopes():
    """
    Compute and store the vector scope graphs for images
    :return: void
    """
    from glob import glob
    import os
    files = glob(INPUT_FOLDER + '*.jpg', recursive=False)
    files.sort()

    for output_folder in [OUTPUT_FOLDER_A, OUTPUT_FOLDER_B]:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    for file in files:
        img = cv2.imread(file)[:, :, ::-1]
        vs_color, vs_gray = compute_vectorscope_base(img, mode='separate', dsize=(256, 256))
        cv2.imwrite(OUTPUT_FOLDER_A + os.path.basename(file), vs_color[:, :, ::-1])
        cv2.imwrite(OUTPUT_FOLDER_B + os.path.basename(file), cv2.cvtColor(vs_gray, cv2.COLOR_GRAY2RGB))


if __name__ == "__main__":
    # image = cv2.imread(FILE_PATH)[:, :, ::-1]  # read image in r-g-b order
    # # zone_system_features = compute_zone_system_features(image)
    # vectorscope = compute_vectorscope_base(image, mode='both')
    # vectorscopes = compute_vectorscope_features(image[np.newaxis, ...], dsize=(64, 64))
    # # util.display_image(vector_scope_gray, mode='L')

    output_vectorscopes()

    pass
