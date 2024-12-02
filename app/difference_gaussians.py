
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from function import FeatureFunction

class DifferenceGaussians:
    def __init__(self, root=None):
        self.root = root
    @staticmethod
    def difference_gaussians_algo(image):

        mask_7x7 = np.array([
            [0, 0, -1, -1, -1, 0, 0],
            [0, -2, -3, -3, -3, -2, 0],
            [-1, -3 , 5 ,5 ,5, -3, -1],
            [-1, -3, 5, 16, 5, -3, -1],
            [-1, -3, 5, 5, 5 , -3, -1],
            [0, -2, -3, -3, -3, -2, 0],
            [0, 0, -1, -1, -1, 0, 0],
        ], dtype=np.float32)

        mask_9x9 = np.array([
            [0, 0, 0, -1, -1, -1, 0 , 0, 0],
            [0, -2, -3, -3, -3, -3, -3, -2, 0],
            [0, -3, -2, -1, -1, -1, -2, -3, 0],
            [-1, -3, -1, 9, 9, 9, -1, -3, -1],
            [-1, -3, -1, 9, 19, 9, -1, -3, -1],
            [-1, -3, -1, 9, 9, 9, -1, -3, -1],
            [0, -3, -2, -1, -1, -1, -2, -3, 0],
            [0, -2, -3, -3, -3, -3, -3, -2, 0],
            [0, 0, 0, -1, -1, -1, 0 , 0, 0],
        ], dtype=np.float32)



        blurred1 = cv2.filter2D(image, -1, mask_7x7)
        blurred2 = cv2.filter2D(image, -1, mask_9x9)
        

        dog = blurred1 - blurred2
        return Image.fromarray(dog.astype(np.uint8)), Image.fromarray(blurred1.astype(np.uint8)), Image.fromarray(blurred2.astype(np.uint8))
        # return dog, blurred1, blurred2
    
    