import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Halftoned:
    def __init__(self, root=None):
        self.root = root
    
    @staticmethod
    def error_diffusion_halftoning(image, threshold=128):

        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape

        for i in range(height):
            for j in range(width):
                old_pixel = img_array[i, j]
                new_pixel = 255 if old_pixel >= threshold else 0  
                img_array[i, j] = new_pixel 

                error = old_pixel - new_pixel

                if j + 1 < width:
                    img_array[i, j + 1] += error * 7 / 16
                if i + 1 < height and j > 0:
                    img_array[i + 1, j - 1] += error * 3 / 16
                if i + 1 < height:
                    img_array[i + 1, j] += error * 5 / 16
                if i + 1 < height and j + 1 < width:
                    img_array[i + 1, j + 1] += error * 1 / 16

        
        
        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    

    

    




