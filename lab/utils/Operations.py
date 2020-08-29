import numpy as np
from scipy import misc
import tensorflow as tf

# 1: Random rotate 2: Random crop  4: Random flip  8:  Fixed image standardization  16: Flip
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16


def _random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')


def _get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)


def create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder):
    """
    Modified `src.facenet.create_input_pipeline`.
    Add image resize function if input image is smaller than 160X160.

    :param input_queue: input_queue
    :param image_size: image_size tuple
    :param nrof_preprocess_threads: nrof_preprocess_threads
    :param batch_size_placeholder: batch_size_placeholder
    :return: image_batch & label_batch
    """
    with tf.name_scope("tempscope"):
        images_and_labels_list = []
        for _ in range(nrof_preprocess_threads):
            filenames, label, control = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)

                # TF 1.2 decode_png will decode both jpg as well as png and vice versa.
                # https://github.com/tensorflow/tensorflow/issues/8551#issuecomment-312746634
                image = tf.image.decode_png(file_contents, 3)

                # Upsampling image if input image is smaller than 160X160
                image = tf.image.resize_images(image, image_size,
                                               method=tf.compat.v1.image.ResizeMethod.BICUBIC)

                # dtype of `_random_rotate_image` output changed from `uint8` to `float32`
                image = tf.cond(_get_control_flag(control[0], RANDOM_ROTATE),
                                lambda: tf.py_func(_random_rotate_image, [image], tf.float32),
                                lambda: tf.identity(image))
                image = tf.cond(_get_control_flag(control[0], RANDOM_CROP),
                                lambda: tf.random_crop(image, image_size + (3,)),
                                lambda: tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
                image = tf.cond(_get_control_flag(control[0], RANDOM_FLIP),
                                lambda: tf.image.random_flip_left_right(image),
                                lambda: tf.identity(image))
                image = tf.cond(_get_control_flag(control[0], FIXED_STANDARDIZATION),
                                lambda: (tf.cast(image, tf.float32) - 127.5) / 128.0,
                                lambda: tf.image.per_image_standardization(image))
                image = tf.cond(_get_control_flag(control[0], FLIP),
                                lambda: tf.image.flip_left_right(image),
                                lambda: tf.identity(image))

                # pylint: disable=no-member
                image.set_shape(image_size + (3,))
                images.append(image)
            images_and_labels_list.append([images, label])

        image_batch, label_batch = tf.train.batch_join(
            images_and_labels_list, batch_size=batch_size_placeholder,
            shapes=[image_size + (3,), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * 100,
            allow_smaller_final_batch=True)

        return image_batch, label_batch
