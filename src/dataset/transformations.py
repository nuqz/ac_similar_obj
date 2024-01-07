import cv2
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from dataset.constants import ORIG_W, ORIG_H


def random_transformation(t):
    def _transform(image, label, factor=0.1, *args, **kwargs):
        if tf.random.uniform([]) < factor:
            image, label = t(image, label, *args, **kwargs)
        return image, label

    return _transform


def curry_transformation(t):
    def _curry(*args, **kwargs):
        def _curried(image, label):
            return t(image, label, *args, **kwargs)

        return _curried

    return _curry


def batch_transformation(t):
    def _transform(images, labels):
        return tf.map_fn(lambda x: t(x[0], x[1]), (images, labels))

    return _transform


def drop_alpha_channel(decoded_image, label):
    return decoded_image[:, :, :3], label


def subtract_from_one_column(x, col_index):
    """
    Supplementary function to flip labels
    """
    num_cols = x.shape[-1]
    cols = tf.split(x,  num_or_size_splits=num_cols, axis=-1)
    cols[col_index] = 1 - cols[col_index]
    upd = tf.concat(cols, axis=-1)
    return upd


@curry_transformation
@random_transformation
def flip_lr(image, label):
    return tf.image.flip_left_right(image), subtract_from_one_column(label, 0)


@curry_transformation
@random_transformation
def flip_ud(image, label):
    return tf.image.flip_up_down(image), subtract_from_one_column(label, 1)


@curry_transformation
@random_transformation
def inverse(image, label):
    return 255 - image, label


def replace_channel(image, channel_index, new_values):
    """
    Supplementary function
    """
    num_channels = tf.shape(image)[2]

    def condition(index, _):
        return index < num_channels

    def body(index, updated_tensor):
        channel = tf.cond(
            tf.equal(index, channel_index),
            lambda: new_values,
            lambda: image[:, :, index:index+1]
        )
        updated_tensor = updated_tensor.write(index, channel)
        return index + 1, updated_tensor

    channels = tf.TensorArray(image.dtype, size=num_channels)
    _, result = tf.while_loop(condition, body, [0, channels])

    return tf.squeeze(tf.transpose(result.stack(), perm=[1, 2, 0, 3]))


@curry_transformation
@random_transformation
def remove_random_channel(image, label, seed=None):
    channel_index = tf.random.uniform([], maxval=3, dtype=tf.int32, seed=seed)
    zeros_channel = tf.expand_dims(
        tf.zeros(tf.shape(image)[0:2], dtype=image.dtype), -1)
    image = replace_channel(image, channel_index, zeros_channel)
    return image, label


@curry_transformation
@random_transformation
def random_brightness(image, label, max_delta=0.15, seed=None):
    return tf.image.random_brightness(image, max_delta, seed), label


@curry_transformation
@random_transformation
def random_hue(image, label, max_delta=0.1, seed=None):
    return tf.image.random_hue(image, max_delta, seed), label


@curry_transformation
@random_transformation
def random_contrast(image, label, lower=0.7, upper=1.3, seed=None):
    return tf.image.random_contrast(image, lower, upper, seed), label


@curry_transformation
@random_transformation
def random_saturation(image, label, lower=0.7, upper=1.3, seed=None):
    return tf.image.random_saturation(image, lower, upper, seed), label


@curry_transformation
@random_transformation
def noise(image, label, amp=32):
    noise = tf.random.uniform(tf.shape(image), -amp, amp, dtype=tf.int32)
    image = tf.cast(image, tf.int32) + noise
    image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    return image, label


def new_depthwise_gaussian_kernel(size=5, mean=0, std=2):
    """
    Supplementary function to create kernel used in gaussian blur
    """
    distribution = tfp.distributions.Normal(mean, std)
    values = distribution.prob(tf.range(-size, size+1, dtype=tf.float32))
    kernel = tf.einsum('i,j->ij', values, values)
    kernel = kernel / tf.reduce_sum(kernel)
    return tf.tile(kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])


@curry_transformation
@random_transformation
def gaussian_blur(image, label, kernel):
    src_type = image.dtype
    image = tf.cast(image[tf.newaxis, :, :, :], dtype=tf.float32)
    image = tf.nn.depthwise_conv2d(image, kernel,
                                   strides=[1, 1, 1, 1], padding='SAME')
    image = tf.squeeze(tf.cast(image, dtype=src_type))
    return image, label


def flatten_label(image, label):
    return image, tf.reshape(label, [-1])


@tf.py_function(Tout=tf.uint8)
def _remove_background(img, threshold=228):
    img = img.numpy()
    if not isinstance(threshold, int):
        threshold = threshold.numpy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, baseline = cv2.threshold(gray, threshold, 255, cv2.THRESH_TRUNC)
    _, background = cv2.threshold(
        baseline, threshold-1, 255, cv2.THRESH_BINARY)
    _, foreground = cv2.threshold(
        baseline, threshold-1, 255, cv2.THRESH_BINARY_INV)

    # Update foreground with bitwise_and to extract real foreground
    foreground = cv2.bitwise_and(img, img, mask=foreground)

    # Convert black and white back into 3 channel greyscale
    background = 255 - cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    # Combine the background and foreground to obtain our final image
    return background+foreground


@curry_transformation
@random_transformation
def remove_background(image, label):
    return _remove_background(image, 235), label


lower_gray = np.array([0, 0, 0])
upper_gray = np.array([255, 20, 255])


@tf.py_function(Tout=tf.uint8)
def _remove_shadows(img):
    img = img.numpy()
    original = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(original, original, mask=mask)
    return result


@curry_transformation
@random_transformation
def remove_shadows(image, label):
    return _remove_shadows(image), label


def restore_shape(image, label):
    """
    Utility transformation to restore image shape after calling py_functions.
    """
    image.set_shape([ORIG_H, ORIG_W, 3])
    return image, label


@curry_transformation
@random_transformation
def add_grid_to_image(image, label, min_size=20, max_size=75, max_thickness=7):
    height, width, channels = image.shape

    grid_size = tf.random.uniform([], min_size, max_size, dtype=tf.int32)
    line_thickness = tf.random.uniform([], 1, max_thickness, dtype=tf.int32)

    grid_x = tf.range(width) % grid_size < line_thickness
    grid_y = tf.range(height) % grid_size < line_thickness

    grid = tf.logical_or(tf.expand_dims(grid_x, 0), tf.expand_dims(grid_y, -1))
    grid = tf.tile(tf.expand_dims(grid, -1), [1, 1, channels])
    grid = tf.cast(grid, tf.uint8)

    return image * (1 - grid), label


default_blur_kernel = new_depthwise_gaussian_kernel(5, 0, 2)
f = {'factor': 0.33}

# TODO: fine-tune transformations parameters
default_transformations = [
    drop_alpha_channel,
    remove_background(**f),
    remove_shadows(**f),
    restore_shape,
    flip_lr(**f),
    flip_ud(**f),
    noise(**f),
    add_grid_to_image(**f),
    gaussian_blur(kernel=default_blur_kernel, **f),
    remove_random_channel(**f),
    inverse(**f),
    random_brightness(**f),
    random_contrast(**f),
    random_hue(**f),
    random_saturation(**f),
    restore_shape,
    flatten_label,
]


def apply_transformations(dataset,
                          transformations=default_transformations,
                          batch=128):
    dataset = dataset.batch(batch)

    for t in transformations:
        dataset = dataset.map(batch_transformation(t),
                              num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.unbatch()
