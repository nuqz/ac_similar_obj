import tensorflow as tf

# TODO: subtract_from_one_column


def random_flip(factor=0.5):
    def _flip(image, label):
        if tf.random.uniform([]) < factor:
            image = tf.image.flip_left_right(image)
            label = tf.tensor_scatter_nd_update(
                label,
                [[0, 0], [1, 0]],
                tf.squeeze(1 - tf.slice(label, [0, 0], [2, 1]))
            )

        if tf.random.uniform([]) < factor:
            image = tf.image.flip_up_down(image)
            label = tf.tensor_scatter_nd_update(
                label,
                [[0, 1], [1, 1]],
                tf.squeeze(1 - tf.slice(label, [0, 1], [2, 1]))
            )

        return image, label

    return _flip


def random_to_grayscale(factor=0.25):
    def _to_grayscale(image, label):
        if tf.random.uniform([]) < factor:
            image = tf.image.rgb_to_grayscale(image)
        return image, label

    return _to_grayscale


def random_inverse(factor=0.25):
    def _inverse(image, label):
        if tf.random.uniform([]) < factor:
            image = 255 - image
        return image, label

    return _inverse


def replace_channel(image, channel_index, new_values):
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


def random_channel_remove(factor=0.75):
    def _channel_remove(image, label):
        if tf.random.uniform([]) < factor:
            channel_index = tf.random.uniform([], maxval=3, dtype=tf.int32)
            zeros_channel = tf.expand_dims(
                tf.zeros(tf.shape(image)[0:2], dtype=image.dtype), -1)
            image = replace_channel(image, channel_index, zeros_channel)

        return image, label

    return _channel_remove
