import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2



class Frequencies:
    def __init__(self, root=None):
        self.root = root
    
    @staticmethod
    def high_pass(image):
        mask_3x3_high_pass = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype = np.float32)

        result = cv2.filter2D(image, -1, mask_3x3_high_pass)
        return Image.fromarray(result.astype(np.uint8))
    

    @staticmethod
    def low_pass(image):
        mask_3x3_low_pass = np.array([
            [0, 1/6, 0],
            [1/6, 2/6, 1/6],
            [0, 1/6, 0]
        ], dtype = np.float32)

        result = cv2.filter2D(image, -1, mask_3x3_low_pass)
        return Image.fromarray(result.astype(np.uint8))
    

    @staticmethod
    def median_filter(image):

        img_array = np.array(image)
        output = np.zeros_like(img_array, dtype=np.uint8)
        height, width = img_array.shape


        for i in range(1, height - 1):
            for j in range(1, width -1 ):
                neighborhood = img_array[i-1:i+2, j-1:j+2]
                median_value = np.median(neighborhood)
                
                output[i, j] = median_value
                
        return Image.fromarray(output)
    
