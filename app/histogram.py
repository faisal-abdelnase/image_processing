import numpy as np
from function import FeatureFunction
from PIL import Image


class Histogram:
    def __init__(self, root=None):
        self.root = root

    @staticmethod
    def calculate_histogram(image):
        image = FeatureFunction.convert_image_color(image)

        histogram = [0] * 256
        bins = list(range(256))  
        image_array = np.array(image)
        for pixel in image_array.flatten():
            histogram[pixel] += 1

        return histogram,bins


