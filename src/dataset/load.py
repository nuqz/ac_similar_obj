import os
import pathlib

import tensorflow as tf

ds_root = pathlib.Path('..' + os.sep + 'ac_similar_obj_ds')
dir_sep = tf.constant(os.sep)
challenges_pattern = str(ds_root/'challenges'/'*.png')
solutions_path = tf.constant(str(ds_root / 'solutions'))
solution_ext = tf.constant('.txt')


def load_label(image_file_path):
    image_file_name = tf.strings.split(image_file_path, dir_sep)[-1]
    image_id = tf.strings.split(image_file_name, '.')[0]
    label_file_path = solutions_path + dir_sep + image_id + solution_ext
    label_raw_str = tf.io.read_file(label_file_path)
    label_lines = tf.strings.split(label_raw_str, '\n')[:-1]
    label_numbers_str = tf.strings.split(label_lines, ',')
    label_rag = tf.strings.to_number(label_numbers_str, tf.float32)
    return tf.reshape(label_rag, [2, -1])


def load(image_file_path):
    image_raw = tf.io.read_file(image_file_path)
    label = load_label(image_file_path)
    return (image_raw, label)


def decode_image(raw_image, label):
    decoded_image = tf.io.decode_png(raw_image)
    return (decoded_image, label)
