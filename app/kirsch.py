import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
from function import FeatureFunction


class Kirsch:
    def __init__(self, root=None):
        self.root = root

    @staticmethod
    def kirsh_algo(image):
        kirsch_masks = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # North
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # Northeast
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # East
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # Southeast
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # South
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # Southwest
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # West
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # Northwest
        ]

        image = FeatureFunction.convert_image_color(image)
        image = np.array(image, dtype=np.float32)

        responses = []
        for mask in kirsch_masks:
            response = convolve2d(image, mask, mode='same', boundary='symm')
            responses.append(response)
        
        
        image_kirsch = np.max(np.stack(responses), axis=0)
        image_kirsch = (image_kirsch / image_kirsch.max() * 255).astype(np.uint8)  
        return Image.fromarray(image_kirsch.astype(np.uint8))
    
    