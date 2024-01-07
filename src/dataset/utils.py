import tensorflow as tf

from dataset.constants import ORIG_H, ORIG_W
from config import dataset_log_dir


def new_summary_writer(name='Dataset preview'):
    return tf.summary.create_file_writer(dataset_log_dir, name=name)


def label_to_points(image, label):
    return tf.cast(
        tf.reverse(tf.reshape(label, (2, 2)), [1]) *
        tf.cast(tf.shape(image)[:2], dtype=tf.float32),
        dtype=tf.int32
    )


def block_indices(block_size=3):
    r = tf.range(block_size)
    dx, dy = tf.meshgrid(r, r, indexing='ij')
    return tf.reshape(tf.stack([dx, dy], axis=-1), [1, block_size**2, 2])


def draw_label(image, label, block_size=3):
    half_block = int(block_size / 2)
    points = label_to_points(image, label)
    points = tf.clip_by_value(points - half_block,
                              0,
                              [
                                  image.shape[0] - block_size,
                                  image.shape[1] - block_size
                              ])

    block = points[:, None, :] + block_indices(block_size)
    block = tf.reshape(block, [-1, 2])
    block = tf.clip_by_value(block,
                             0,
                             [
                                 image.shape[0] - 1,
                                 image.shape[1] - 1
                             ])

    mask = tf.tensor_scatter_nd_update(
        image,
        block,
        255 - tf.gather_nd(image, block)
    )

    return mask


def dummy_image(n_batches=1):
    return tf.ones([n_batches, ORIG_H, ORIG_W, 3])
