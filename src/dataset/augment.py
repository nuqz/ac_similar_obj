import tensorflow as tf
import tensorflow_probability as tfp


def subtract_from_one_column(x, col_index):
    num_cols = x.shape[-1]
    cols = tf.split(x,  num_or_size_splits=num_cols, axis=-1)
    cols[col_index] = 1 - cols[col_index]
    upd = tf.concat(cols, axis=-1)
    return upd


def random_flip(factor=0.5):
    def _flip(image, label):
        if tf.random.uniform([]) < factor:
            image = tf.image.flip_left_right(image)
            label = subtract_from_one_column(label, 0)

        if tf.random.uniform([]) < factor:
            image = tf.image.flip_up_down(image)
            label = subtract_from_one_column(label, 1)

        return image, label

    return _flip


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


def random_brightness(max_delta=0.25):
    def _brightness(image, label):
        image = tf.image.random_brightness(image, max_delta)
        return image, label

    return _brightness


def random_contrast(lower=0.25, upper=1.75):
    def _contrast(image, label):
        image = tf.image.random_contrast(image, lower, upper)
        return image, label

    return _contrast


def random_hue(max_delta=0.25):
    def _hue(image, label):
        image = tf.image.random_hue(image, max_delta)
        return image, label

    return _hue


def random_saturation(lower=0.25, upper=1.75):
    def _saturation(image, label):
        image = tf.image.random_saturation(image, lower, upper)
        return image, label

    return _saturation


def random_noise(factor=0.5, amp=32):
    def _noise(image, label):
        print('b', image.shape)
        print('n', tf.shape(image))
        if tf.random.uniform([]) < factor:
            shape = tf.shape(image)
            noise = tf.random.uniform(shape, 0, amp, dtype=tf.int32)
            noise = tf.cast(noise, tf.uint8)
            image = tf.clip_by_value(
                image + noise, clip_value_min=0, clip_value_max=255)
        return image, label

    return _noise


def new_gaussian_kernel_2d(size, mean, std):
    d = tfp.distributions.Normal(mean, std)
    vals = d.prob(tf.range(-size, size+1, dtype=tf.float32))
    kernel = tf.einsum('i,j->ij', vals, vals)
    return kernel / tf.reduce_sum(kernel)


def random_blur(factor=0.5, kernel_size=5, sigma=2):
    def _blur(image, label):
        if tf.random.uniform([]) < factor:
            kernel = new_gaussian_kernel_2d(kernel_size, 0, sigma)
            kernel = kernel[:, :, tf.newaxis, tf.newaxis]
            kernel = tf.tile(kernel, [1, 1, 3, 1])

            image = tf.cast(image[tf.newaxis, :, :, :], dtype=tf.float32)
            image = tf.nn.depthwise_conv2d(
                image, kernel, strides=[1, 1, 1, 1], padding='SAME')
            image = tf.squeeze(tf.cast(image, dtype=tf.uint8))

        return image, label

    return _blur
