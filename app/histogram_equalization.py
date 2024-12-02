import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from function import FeatureFunction
from histogram import Histogram

class HistogramEqualization:
    def __init__(self, root=None):
        self.root = root

    @staticmethod
    def histogram_equalization(image):
        image = FeatureFunction.convert_image_color(image)
        histogram, bins = Histogram.calculate_histogram(image)

        cdf = np.cumsum(histogram)
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min()) 

        img_array = np.array(image)
        flat = img_array.flatten()

        equalized_flat = np.interp(flat, bins, cdf_normalized).astype(np.uint8)
        equalized_img_array = equalized_flat.reshape(img_array.shape)
        equalized_image = Image.fromarray(equalized_img_array)
        # equalized_histogram, _ = Histogram.calculate_histogram(equalized_image)

        return equalized_image
        
        
    
    
