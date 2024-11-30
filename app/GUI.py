import tkinter as tk
from tkinter import messagebox
from PIL import Image
import matplotlib.pyplot as plt

from halftoning import Halftoned
from histogram import Histogram
from sobel import Sobel
from prewitt import Prewitt
from homogeneity import Homogeneity
from difference import Difference
from difference_gaussians import DifferenceGaussians
from kirsch import Kirsch





# Define a class with a method to be triggered
class GUI:
    def __init__(self, root):
        self.root = root

        # Create a button and bind it to the class method
        halftoningButton = tk.Button(root, text="Halftoned", command = self.halftonedAlg)
        halftoningButton.grid(row=0, column=0,pady=10, padx=10)

        histogramButton = tk.Button(root, text="Histogram", command = self.histogramAlg)
        histogramButton.grid(row=0, column=1,pady=10, padx=10)

        sobelButton = tk.Button(root, text="Sobel", command = self.sobelAlg)
        sobelButton.grid(row=0, column=2,pady=10, padx=10)

        prewittButton = tk.Button(root, text="Prewitt", command = self.prewittAlg)
        prewittButton.grid(row=1, column=0,pady=10, padx=10)

        kirschButton = tk.Button(root, text="Kirsch", command = self.kirsch)
        kirschButton.grid(row=1, column=1,pady=10, padx=10)

        homogeneityButton = tk.Button(root, text="Homogeneity", command = self.homogeneity)
        homogeneityButton.grid(row=1, column=2,pady=10, padx=10)

        differenceButton = tk.Button(root, text="Difference", command = self.difference)
        differenceButton.grid(row=2, column=0,pady=10, padx=10)

        differenceGauButton = tk.Button(root, text="Difference Gaussians", command = self.difference_gaussian)
        differenceGauButton.grid(row=2, column=1,pady=10, padx=10)


        


    def halftonedAlg(self):
        Halftoned.showResultHalftoned()

    def histogramAlg(self):
        Histogram.showResultHistogram()

    def sobelAlg(self):
        Sobel.showResultSobel()

    def prewittAlg(self):
        Prewitt.showResultPrewitt()

    def kirsch(self):
        Kirsch.showResultKirsch()

    def homogeneity(self):
        Homogeneity.showResultHomogeneity()

    def difference(self):
        Difference.showResultDifference()

    def difference_gaussian(self):
        DifferenceGaussians.showResultDifferenceGaussian()

    
        


# Main application loop
if __name__ == "__main__":
    root = tk.Tk()  # Create the main window
    app = GUI(root)  # Instantiate the class
    root.mainloop()  # Run the GUI event loop
