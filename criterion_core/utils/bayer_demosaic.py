import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np

kernels = {
    "bgr": np.array([[[1, 0, 0], [0, 0.5, 0]], [[0, 0.5, 0], [0, 0, 1]]]),
    "rgb": np.array([[[0, 0, 1], [0, 0.5, 0]], [[0, 0.5, 0], [1, 0, 0]]]),
}

def BayerDemosaic(mode="rgb"):
    kernel = kernels[mode.lower()]
    return Conv2D(3, kernel.shape[:2],
                  use_bias=False,
                  strides=2,
                  weights=[kernel[:,:,None,:]],
                  trainable=False,
                  padding='same')


def tf_bayer_demosaic(x, mode="rgb"):
    return tf.nn.conv2d(x, kernels["rgb"][:,:,None,:], strides=[1, 2, 2, 1], padding='SAME')