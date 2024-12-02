import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from function import FeatureFunction


class Sobel:
    def __init__(self, root=None):
        self.root = root
    @staticmethod
    def apply_sobel_edge_detection(image):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        
        sobel_y = np.array([[ 1,  2,  1],
                            [ 0,  0,  0],
                            [-1, -2, -1]])

        image = FeatureFunction.convert_image_color(image)


        gray_image = np.array(image, dtype=float)
        
        h, w = gray_image.shape
        
        
        gradient_x = np.zeros_like(gray_image)
        gradient_y = np.zeros_like(gray_image)
        
        # Pad the image to handle borders
        padded_image = np.pad(gray_image, pad_width=1, mode='constant', constant_values=0)
        
        
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                region = padded_image[i-1:i+2, j-1:j+2]
            
                gradient_x[i-1, j-1] = np.sum(sobel_x * region)
                gradient_y[i-1, j-1] = np.sum(sobel_y * region)
        
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
        
        return Image.fromarray(gradient_magnitude.astype(np.uint8))
    

