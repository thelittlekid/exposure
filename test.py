from PIL import Image
import cv2
import numpy as np

FILE_PATH = 'data/artists/FiveK_C/0001.jpg'
FILE_PATH = 'data/artists/Ansel_Adams/Afternoon sun.jpg'

image = cv2.imread(FILE_PATH)[:, :, ::-1]  # read image in r-g-b order

img = Image.fromarray(image, 'RGB')
img.show()

# data = np.load('train_image.npy')
#
# image = np.uint8(data[0, :, :, :] * 255)
# img = Image.fromarray(image, 'RGB')
# img.show()


pass