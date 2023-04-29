import os

import numpy as np
import tensorflow as tf


def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg




from PIL import Image
#Open image using Image module
im = Image.open("D:/LjmuMSc/Projects/Github/Relevent_repos/CIHP_PGN-master/CIHP_PGN-master/datasets/CIHP/images/0002190.jpg")
#Show actual Image
# im.show()
# #Show rotated Image
# im = im.rotate(45)
# im.show()

# value_1 = my_func(im)
# print(value_1)

new = Image.new(mode="RGB", size=(im.size))

img_contents = [new for x in range(10)]

img = tf.image.decode_jpeg(img_contents, channels=3)

print(img)