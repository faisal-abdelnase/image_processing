import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

class VarianceAndRange:
    def __init__(self, root=None):
        self.root = root
    
    @staticmethod
    def variance_operator(image):
        img_array = np.array(image)

        output = np.zeros_like(img_array, dtype=np.uint8)
        height, width = img_array.shape


        for i in range(1, height - 1):
            for j in range(1, width -1 ):
                neighborhood = img_array[i-1:i+2, j-1:j+2]
                mean = np.mean(neighborhood)
                Variance = np.sum((neighborhood - mean) ** 2) / 9
                output[i, j] = Variance
                
        return Image.fromarray(output)
            
    @staticmethod
    def range_operator(image):
        img_array = np.array(image)
        output = np.zeros_like(img_array, dtype=np.uint8)
        height, width = img_array.shape

        for i in range(1, height - 1):
            for j in range(1, width -1 ):
                neighborhood = img_array[i-1:i+2, j-1:j+2]
                range_value = np.max(neighborhood) - np.min(neighborhood)
                output[i, j] = range_value
        return Image.fromarray(output)
            



