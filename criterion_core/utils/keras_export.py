import logging
import os
import tarfile

# if os.name == "nt":
# Windows hack to fix error in generic_utils
# from tensorflow.python.keras.utils import generic_utils
# utils_path = generic_utils.__file__
# patch with file criterion_core/utils/export_model_generic_utils.patch
import tensorflow as tf
from tensorflow import keras

from criterion_core.utils import bayer_demosaic
from .path import movedir

log = logging.getLogger(__name__)

decoder_map = {
    "jpeg": tf.image.decode_jpeg,
    "jpeg": tf.image.decode_jpeg,
    "bmp": tf.image.decode_bmp,
    "png": tf.image.decode_png,
}


def define_image_input_receiver(input_tensor, img_format, input_shape, color_mode):
    """
    Image loading pipeline
    Loads and scales image to range [0,1) t
    :param input_tensor: Name of input tensor
    :param img_format:
    :param input_shape:
    :param color_mode: color mode (set to bayer if the input should be bayer filtered)
    :return:
    """
    log.info("Building input receiver fn")
    def serving_input_receiver_fn():
        def prepare_image(image_str_tensor):
            channels = 1 if color_mode == "bayer" else input_shape[2]
            log.info("Channels: {}".format(channels))
            image = decoder_map[img_format](image_str_tensor, channels=channels)
            return image

        input_ph = tf.placeholder(tf.string, shape=[None])
        images = tf.map_fn(prepare_image, input_ph, back_prop=False, dtype=tf.uint8)
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)

        if color_mode == "bayer":
            log.info("Applying Bayer filtering")
            images = bayer_demosaic.tf_bayer_demosaic(images, "rgb")

        log.info("Resize input image to: {}".format(input_shape[:2]))
        images = tf.image.resize_bilinear(images, input_shape[:2], align_corners=False)

        return tf.estimator.export.ServingInputReceiver(
            {input_tensor: images},
            {'image_bytes': input_ph})

    return serving_input_receiver_fn


def export_model(keras_model, img_format, input_shape, color_mode, export_path='tf_export', tf_model_path='tf_new'):
    estimator = keras.estimator.model_to_estimator(
        model_dir=tf_model_path,
        keras_model=keras_model)

    movedir(os.path.join(tf_model_path, 'keras'), tf_model_path)

    export_path_output = estimator.export_savedmodel(
        export_path, checkpoint_path=os.path.join(tf_model_path, "keras_model.ckpt"),
        serving_input_receiver_fn=define_image_input_receiver(keras_model.input.name.split(":", 1)[0], img_format,
                                                              input_shape, color_mode))
    export_path_output = str(export_path_output, encoding="utf-8")
    saved_model_tar = os.path.join(export_path_output, 'saved_model.tar.gz')
    with tarfile.open(saved_model_tar, "w:gz") as tar:
        tar.add(export_path_output, arcname="saved_model")
    return saved_model_tar
