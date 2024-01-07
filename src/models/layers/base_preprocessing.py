import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Resizing, Rescaling


class BasePreprocessing(Layer):
    def __init__(self, new_size, normalization_scale=1./255, grayscale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h, w = new_size
        self.resizing = Resizing(h, w)
        self.normalization = Rescaling(normalization_scale)
        self.grayscale = grayscale

    def call(self, input):
        x = self.resizing(input)
        x = self.normalization(x)

        if self.grayscale:
            x = tf.image.rgb_to_grayscale(x)

        return x
