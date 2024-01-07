from keras.layers import Input

from dataset import ORIG_H, ORIG_W
from models.layers import BasePreprocessing


def build_head(shrink=6, grayscale=True):
    input = Input(shape=(ORIG_H, ORIG_W, 3))
    x = BasePreprocessing((ORIG_H // shrink, ORIG_W // shrink),
                          grayscale=grayscale)(input)

    return input, x
