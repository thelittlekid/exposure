import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

FILE_PATH = 'data/artists/FiveK_C/0001.jpg'


def compute_zone_system_features(img, zone_num=11, mode='mask'):
    """
    Compute additional features on color distribution using the "Zone System" proposed by Ansel Adams and Fred Archer
    Each feature map contains pixels of a particular color in a particular zone. The feature map has the same height and
    width as the original input image.

    :param img: the original input image (h x w x d) or the tensor holding the image (? x h x w x d)
    :param zone_num: total number of zones, 11 by default
    :param mode: 'mask' - using binary mask by default; 'color' (or else) - use color map
    :return: zone-system features - numpy array with shape h x w x (zone_num+d)
            or a tensor with shape (? x h x w x (zone_num+d))
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


if __name__ == "__main__":
    image = cv2.imread(FILE_PATH)[:, :, ::-1]  # read image in r-g-b order
    zone_system_features = compute_zone_system_features(image)

    pass
