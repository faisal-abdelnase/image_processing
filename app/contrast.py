
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

class Contrast:
    def __init__(self, root=None):
        self.root = root
    
    @staticmethod
    def contrast_based_edge_detection(image):
        edge_mask = np.array([
            [-1, 0, -1],
            [0, 4, 0],
            [-1, 0, -1]
        ])

        smoothing_mask = np.ones((3,3)) / 9
        edge_output = cv2.filter2D(image, -1, edge_mask)

        average_output = cv2.filter2D(image, -1, smoothing_mask)
        average_output = average_output.astype(float)
        average_output += 1e-10

        contrast_edges = edge_output / average_output

        # return contrast_edges, edge_output, average_output
        return Image.fromarray(contrast_edges.astype(np.uint8)), Image.fromarray(edge_output.astype(np.uint8)), Image.fromarray(average_output.astype(np.uint8))
    
    
