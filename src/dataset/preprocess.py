import cv2
import numpy as np
import tensorflow as tf

ORIG_W, ORIG_H = 552, 344
RESIZE_FACTOR = 0.5
RESCALE_FACTOR = 1./255
W, H = int(RESIZE_FACTOR * ORIG_W), int(RESIZE_FACTOR * ORIG_H)
CROP_W, CROP_H = 30, 15



def drop_alpha_channel(decoded_image, label):
    return (decoded_image[:, :, :3], label)


def normalize_image(image, label):
def _remove_background(img, threshold=228):
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


def _remove_shadows(img):
    original = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(original, original, mask=mask)
    return result


@tf.py_function(Tout=(tf.uint8, tf.float32))
def remove_background(image, label):
    return _remove_background(image.numpy(), 235), label


@tf.py_function(Tout=(tf.uint8, tf.float32))
def remove_shadows(image, label):
    return _remove_shadows(image.numpy()), label
