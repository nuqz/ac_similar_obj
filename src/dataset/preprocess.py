import cv2
import numpy as np
import tensorflow as tf

from dataset.constants import ORIG_W, ORIG_H


def drop_alpha_channel(decoded_image, label):
    return (decoded_image[:, :, :3], label)


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


def remove_background(image, label):
    return _remove_background(image, 235), label


def remove_shadows(image, label):
    return _remove_shadows(image), label
