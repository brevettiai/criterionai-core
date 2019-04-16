import tensorflow as tf
from tensorflow import keras
from .path import movedir
import tarfile
import os

def define_image_input_receiver(input_tensor, img_format, input_shape):
    """
    Image loading pipeline
    Loads and scales image to range [0,1) t
    :param input_tensor: Name of input tensor
    :param img_format:
    :param input_shape:
    :return:
    """
    def serving_input_receiver_fn():
        def prepare_image(image_str_tensor):
            print("Input shape:", input_shape, input_shape[-1])
            if img_format=='bmp':
                image = tf.image.decode_bmp(image_str_tensor, channels=input_shape[-1])
            if img_format=='jpeg' or img_format=='jpg':
                image = tf.image.decode_jpeg(image_str_tensor, channels=input_shape[-1])
            if img_format=='png':
                image = tf.image.decode_png(image_str_tensor, channels=input_shape[-1])
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, input_shape[:2], align_corners=False)
            image = tf.squeeze(image, axis=[0])
            image = tf.cast(image, dtype=tf.uint8)
            return image
        input_ph = tf.placeholder(tf.string, shape=[None])
        images_tensor = tf.map_fn(
            prepare_image, input_ph, back_prop=False, dtype=tf.uint8)
        images_tensor = tf.image.convert_image_dtype(images_tensor, dtype=tf.float32)
        return tf.estimator.export.ServingInputReceiver(
            {input_tensor: images_tensor},
            {'image_bytes': input_ph})
    return serving_input_receiver_fn


def export_model(keras_model, img_format, input_shape, export_path='tf_export', tf_model_path='tf_new'):
    estimator = keras.estimator.model_to_estimator(
        model_dir=tf_model_path,
        keras_model=keras_model)

    movedir(os.path.join(os.getcwd(), tf_model_path, 'keras'), os.path.join(os.getcwd(), tf_model_path))

    estimator.export_savedmodel(
        export_path,
        serving_input_receiver_fn=define_image_input_receiver(keras_model.input.name.split(":",1)[0], img_format, input_shape))
    all_exports = sorted(filter(lambda x: not x.startswith("temp-"), os.listdir(export_path)), key=lambda x: int(x))
    newest = all_exports[-1]

    export_path_newest = os.path.join(export_path, newest)
    saved_model_tar = os.path.join(export_path_newest, 'saved_model.tar.gz')
    with tarfile.open(saved_model_tar, "w:gz") as tar:
        tar.add(export_path_newest, arcname="saved_model")
    return saved_model_tar