import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Resizing, Rescaling


class BasePreprocessing(Layer):
    def __init__(self, new_size, normalization_scale=1./255, *args, **kwargs):
        super(BasePreprocessing, self).__init__(*args, **kwargs)
        h, w = new_size
        self.resizing = Resizing(h, w)
        self.normalization = Rescaling(normalization_scale)

    def call(self, input):
        x = self.resizing(input)
        x = self.normalization(x)
        return x
