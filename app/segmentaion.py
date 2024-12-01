import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from function import FeatureFunction
from scipy.signal import find_peaks

class Segmentaion:
    def __init__(self, root=None):
        self.root = root
    
    @staticmethod
    def manual_segmentation(image, low_threshold, high_threshold, value=255):
        image = FeatureFunction.convert_image_color(image)
        image_array = np.array(image)
        segmentation_image = np.zeros_like(image_array)
        
        segmentation_image[(image_array >= low_threshold) & (image_array <= high_threshold)] = value
        
        return Image.fromarray(segmentation_image)
    



    @staticmethod
    def histogram_peak_threshold_segmentation(image):

        image = FeatureFunction.convert_image_color(image)
        image_array = np.array(image)
        
        hist = cv2.calcHist([image_array], [0], None, [256], [0, 255]).flatten()
        peaks_indices = FeatureFunction.find_histogram_peaks(hist)
        low_threshold, high_threshold = FeatureFunction.calculate_thresholds_peak(peaks_indices, hist)
        print([low_threshold, high_threshold])
        # image_array = np.array(image)
        segmented_image = np.zeros_like(image_array)
        segmented_image[(image_array >= low_threshold) & (image_array <= high_threshold)] = 255

        return Image.fromarray(segmented_image)
    


    
    @staticmethod
    def histogram_valley_threshold_segmentation(image):

        image = FeatureFunction.convert_image_color(image)
        image_array = np.array(image)
        
        hist = cv2.calcHist([image_array], [0], None, [256], [0, 255]).flatten()
        peaks_indices = FeatureFunction.find_histogram_peaks(hist)
        valley_point = FeatureFunction.find_valley_point(peaks_indices, hist)
        low_threshold, high_threshold = FeatureFunction.calculate_thresholds_valley(peaks_indices, valley_point)
        print([low_threshold, high_threshold])
        
        segmented_image = np.zeros_like(image_array)
        segmented_image[(image_array >= low_threshold) & (image_array <= high_threshold)] = 255

        return Image.fromarray(segmented_image)
    





    @staticmethod
    def adaptive_histogram_threshold_segmentation(image):

        image = FeatureFunction.convert_image_color(image)
        image_array = np.array(image)
        
        hist = cv2.calcHist([image_array], [0], None, [256], [0, 255]).flatten()
        peaks_indices = FeatureFunction.find_histogram_peaks(hist)
        valley_point = FeatureFunction.find_valley_point(peaks_indices, hist)
        low_threshold, high_threshold = FeatureFunction.calculate_thresholds_valley(peaks_indices, valley_point)
        print([low_threshold, high_threshold])
        
        segmented_image = np.zeros_like(image_array)
        segmented_image[(image_array >= low_threshold) & (image_array <= high_threshold)] = 255

        background_mean, object_mean = FeatureFunction.calculat_means(segmented_image, image_array)
        new_peaks_indices = [int(background_mean), int(object_mean)]
        new_valley_point = FeatureFunction.find_valley_point(new_peaks_indices, hist)
        new_low_threshold, new_high_threshold = FeatureFunction.calculate_thresholds_valley(new_peaks_indices, new_valley_point)

        print([new_low_threshold, new_high_threshold])

        final_segmented_image = np.zeros_like(image_array)
        final_segmented_image[(image_array >= new_low_threshold) & (image_array <= new_high_threshold)] = 255

        return Image.fromarray(segmented_image)