import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from app.function import FeatureFunction

class ImageOperation:
    def __init__(self, root=None):
        self.root = root
    
    @staticmethod
    def add_operation(image):

        image = FeatureFunction.convert_image_color(image)

        image1 = np.array(image)
        image2 = np.array(image)

        height, width = image1.shape
        added_image = np.zeros_like(image2, dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                added_image[i, j] = image1[i, j] + image2[i, j]
                added_image[i, j] = max(0, min(added_image[i, j], 255))

        return Image.fromarray(added_image)
    

    @staticmethod
    def subtract_operation(image):

        image = FeatureFunction.convert_image_color(image)

        image1 = np.array(image)
        image2 = np.array(image)

        height, width = image1.shape
        subtracted_image = np.zeros_like(image2, dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                subtracted_image[i, j] = image1[i, j] - image2[i, j]
                subtracted_image[i, j] = max(0, min(subtracted_image[i, j], 255))

        return Image.fromarray(subtracted_image)
    





    @staticmethod
    def invert_operation(image):

        image = FeatureFunction.convert_image_color(image)
        image_array = np.array(image)

        height, width = image_array.shape
        inverted_image = np.zeros_like(image_array, dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                inverted_image[i, j] = 255 - image_array[i, j]
                

        return Image.fromarray(inverted_image)
