
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from app.function import FeatureFunction



class Homogeneity:
    def __init__(self, root=None):
        self.root = root
    @staticmethod
    def homogeneity_algo(image , threshold=5):
        image = FeatureFunction.convert_image_color(image)
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape
        homogeneity_image = np.zeros_like(img_array)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center_pixel = img_array[i,j]
                differences = [
                    abs(center_pixel - img_array[i-1, j-1]),
                    abs(center_pixel - img_array[i-1, j]),
                    abs(center_pixel - img_array[i-1, j+1]),
                    abs(center_pixel - img_array[i, j-1]),
                    abs(center_pixel - img_array[i, j+1]),
                    abs(center_pixel - img_array[i+1, j-1]),
                    abs(center_pixel - img_array[i+1, j]),
                    abs(center_pixel - img_array[i+1, j+1]),
                ]

                homogeneity_value = max(differences)
                homogeneity_image[i, j] = homogeneity_value
                homogeneity_image[i,j] = np.where(homogeneity_image[i,j] >= threshold, homogeneity_image[i,j], 0)
                

        return Image.fromarray(homogeneity_image.astype(np.uint8))

    