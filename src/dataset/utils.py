import tensorflow as tf

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


def split_dataset(dataset, train_size=0.8, seed=None, shuffle_buffer=None):
    if seed is None:
        seed = tf.random.uniform([], maxval=255, dtype=tf.int64)

    dataset_length = dataset.cardinality().numpy()
    if shuffle_buffer is None:
        shuffle_buffer = dataset_length

    train_length = int(dataset_length * train_size)
    train_ds = dataset.shuffle(shuffle_buffer, seed=seed) \
        .take(train_length)
    validation_ds = dataset.shuffle(shuffle_buffer, seed=seed) \
        .skip(train_length)

    return train_ds, validation_ds


def configure_for_performance(dataset, shuffle_buffer=None, batch_size=128):
    if shuffle_buffer is None:
        shuffle_buffer = dataset.cardinality()

    dataset = dataset.cache()
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
