import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from app.function import FeatureFunction

class Difference:
    def __init__(self, root=None):
        self.root = root
    @staticmethod
    def difference_algo(image , threshold=5):
        image = FeatureFunction.convert_image_color(image)
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape
        difference_image = np.zeros_like(img_array)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                diff1 = abs(img_array[i-1, j-1] - img_array[i+1, j+1])
                diff2 = abs(img_array[i-1, j+1] - img_array[i+1, j-1]) 
                diff3 = abs(img_array[i, j-1] - img_array[i, j+1]) 
                diff4 = abs(img_array[i-1, j] - img_array[i+1, j]) 

                max_diff = max(diff1, diff2, diff3, diff4)
                difference_image[i,j] = max_diff

                difference_image[i,j] = np.where(difference_image[i,j] >= threshold, difference_image[i,j], 0)

        return Image.fromarray(difference_image.astype(np.uint8))
    
    
