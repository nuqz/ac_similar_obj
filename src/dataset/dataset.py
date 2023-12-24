import os
import pathlib

import cv2
import tensorflow as tf

from dataset.load import challenges_pattern, load, decode_image
from dataset.preprocess import drop_alpha_channel, \
    remove_background, remove_shadows, restore_shape
from dataset.augment import random_flip, random_inverse, random_channel_remove, \
    random_brightness, random_contrast, random_hue, random_saturation, \
    random_noise, random_blur


loaded_and_decoded_ds = tf.data.Dataset.list_files(challenges_pattern) \
    .map(load) \
    .map(decode_image) \
    .map(drop_alpha_channel)

without_bg_ds = loaded_and_decoded_ds.map(remove_background)
without_bg_and_shadows_ds = without_bg_ds.map(remove_shadows)

triple_ds = loaded_and_decoded_ds \
    .concatenate(without_bg_ds) \
    .concatenate(without_bg_and_shadows_ds)

augmented_ds = triple_ds.repeat(2) \
    .map(restore_shape) \
    .map(random_brightness(max_delta=0.1), num_parallel_calls=tf.data.AUTOTUNE) \
    .map(random_contrast(lower=0.7, upper=1.3), num_parallel_calls=tf.data.AUTOTUNE) \
    .map(random_hue(max_delta=0.1), num_parallel_calls=tf.data.AUTOTUNE) \
    .map(random_saturation(lower=0.7, upper=1.3), num_parallel_calls=tf.data.AUTOTUNE) \
    .map(random_channel_remove(), num_parallel_calls=tf.data.AUTOTUNE) \
    .map(random_inverse(), num_parallel_calls=tf.data.AUTOTUNE) \
    .map(random_flip(), num_parallel_calls=tf.data.AUTOTUNE) \
    .map(random_blur(), num_parallel_calls=tf.data.AUTOTUNE) \
    .map(random_noise(0.5), num_parallel_calls=tf.data.AUTOTUNE) \
    .map(lambda image, label: (image, tf.reshape(label, [-1])), num_parallel_calls=tf.data.AUTOTUNE) \
    .map(restore_shape) \
