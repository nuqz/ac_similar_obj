import os
import pathlib

import cv2
import tensorflow as tf

from dataset.load import challenges_pattern, load, decode_image
from dataset.preprocess import drop_alpha_channel, \
    remove_background, remove_shadows
from dataset.augment import random_flip, random_to_grayscale, random_inverse, random_channel_remove


loaded_and_decoded_ds = tf.data.Dataset.list_files(challenges_pattern) \
    .map(load) \
    .map(decode_image) \
    .map(drop_alpha_channel)

without_bg_ds = loaded_and_decoded_ds.map(remove_background)
without_bg_and_shadows_ds = without_bg_ds.map(remove_shadows)

triple_ds = loaded_and_decoded_ds \
    .concatenate(without_bg_ds) \
    .concatenate(without_bg_and_shadows_ds)

# Augmentations init
_random_flip = random_flip()
_random_to_grayscale = random_to_grayscale()
_random_inverse = random_inverse()
_random_channel_remove = random_channel_remove()

# .map(lambda image, label: _random_flip(image, label))
# .map(_random_to_grayscale)
# .map(_random_inverse)
augmented_ds = triple_ds \
    .map(_random_channel_remove)
