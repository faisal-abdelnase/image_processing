import cv2
import numpy as np
from scipy.signal import find_peaks

class FeatureFunction:
    def __init__(self, root):
        self.root = root

    def convert_image_color(image):
        if image.mode != 'L':
            image_gray = image.convert('L')
            return image_gray
        else:
            return image
        
    def calculates_threshold(image):
        average_value = np.mean(image)

        return average_value
    

    def find_histogram_peaks(hist):
        peaks, _ = find_peaks(hist, height = 0)
        sorted_peaks = sorted(peaks, key=lambda x : hist[x], reverse= True)
        return sorted_peaks[: 2]
    

    def calculate_thresholds_peak(peaks_indices, hist):
        peak1 = peaks_indices[0]
        peak2 = peaks_indices[1]

        low_threshold = (peak1 + peak2) // 2
        high_threshold = peak2
        return low_threshold, high_threshold
    

    def find_valley_point(peaks_indices, hist):
        valley_point = 0
        min_valley = float('inf')
        start, end = peaks_indices
        for i in range(start, end + 1):
            if hist[i] < min_valley:
                min_valley = hist[i]
                valley_point = i

        return valley_point
        

    def calculate_thresholds_valley(peaks_indices, valley_point):
        low_threshold = valley_point
        high_threshold = peaks_indices[1]

        return low_threshold, high_threshold
    

    def calculat_means(segmented_image, image_array):
        object_pixels = image_array[segmented_image == 255]
        background_pixels = image_array[segmented_image == 0]

        object_mean = object_pixels.mean() if object_pixels.size > 0 else 0
        background_mean = background_pixels.mean() if background_pixels.size > 0 else 0

        return background_mean, object_mean


