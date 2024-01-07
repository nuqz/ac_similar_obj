import tensorflow as tf

from dataset.load import challenges_pattern, load, decode_image

loaded_and_decoded_ds = tf.data.Dataset.list_files(challenges_pattern) \
    .map(load, num_parallel_calls=tf.data.AUTOTUNE) \
    .map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)


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
