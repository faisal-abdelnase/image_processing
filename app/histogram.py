import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from app.function import FeatureFunction

class Histogram:
    def __init__(self, root=None):
        self.root = root

    @staticmethod
    def histogram_equalization(image):
        
        image = FeatureFunction.convert_image_color(image)

        
        img_array = np.array(image)
        flat = img_array.flatten()
        
        hist, bins = np.histogram(flat, bins=256, range=[0, 256], density=True)
        
        cdf = hist.cumsum()
        cdf_normalized = cdf * (255 / cdf[-1])
        
        equalized_img_array = np.interp(flat, bins[:-1], cdf_normalized).reshape(img_array.shape)
        
        return Image.fromarray(equalized_img_array.astype(np.uint8))
    
    
    
    
