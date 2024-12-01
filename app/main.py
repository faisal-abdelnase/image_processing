from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from halftoning import Halftoned
from function import FeatureFunction
from histogram import Histogram
from sobel import Sobel
from prewitt import Prewitt
from kirsch import Kirsch
from homogeneity import Homogeneity
from difference import Difference
from difference_gaussians import DifferenceGaussians
from contrast import Contrast
from variance import VarianceAndRange
from frequencies import Frequencies
from image_operations import ImageOperation
from segmentaion import Segmentaion
from PIL import Image
from io import BytesIO
import os
import cv2


app = Flask(__name__)


# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploaded_images")
TEMP_IMAGE_DIR = os.path.join(BASE_DIR, "processed_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TEMP_IMAGE_DIR"] = TEMP_IMAGE_DIR



def halftonedAlg(image_path):
    image = Image.open(image_path)
    image = FeatureFunction.convert_image_color(image)
    img = Halftoned().error_diffusion_halftoning(image)  
    filename = "halftoned.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename


def histogramAlg(image_path):
    image = Image.open(image_path)
    img = Histogram().histogram_equalization(image)
    filename = "histogram.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename


def sobelAlg(image_path):
    image = Image.open(image_path)
    img = Sobel().apply_sobel_edge_detection(image)
    filename = "sobel.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename

def prewittAlg(image_path):
    image = Image.open(image_path)
    img = Prewitt().apply_prewitt_edge_detection(image)
    filename = "prewitt.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename

def kirschAlg(image_path):
    image = Image.open(image_path)
    img = Kirsch().kirsh_algo(image)
    filename = "kirsch.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename

def homogeneityAlg(image_path):
    image = Image.open(image_path)
    threshold = FeatureFunction.calculates_threshold(image)
    img = Homogeneity().homogeneity_algo(image, threshold)
    filename = "homogeneity.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename

def differenceAlg(image_path):
    image = Image.open(image_path)
    threshold = FeatureFunction.calculates_threshold(image)
    img = Difference().difference_algo(image, threshold)
    filename = "difference.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename



def differenceGaussianAlg(image_path):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    dog_image, blurred1_image, blurred2_image = DifferenceGaussians().difference_gaussians_algo(image)
    
    dog_filename = "difference_gaussian_dog.png"
    blurred1_filename = "difference_gaussian_blurred1.png"
    blurred2_filename = "difference_gaussian_blurred2.png"
    
    dog_path = os.path.join(TEMP_IMAGE_DIR, dog_filename)
    blurred1_path = os.path.join(TEMP_IMAGE_DIR, blurred1_filename)
    blurred2_path = os.path.join(TEMP_IMAGE_DIR, blurred2_filename)
    
    dog_image.save(dog_path)
    blurred1_image.save(blurred1_path)
    blurred2_image.save(blurred2_path)
    
    return [dog_filename, blurred1_filename, blurred2_filename]



def contrastAlgo(image_path):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contrast_edges, edge_output, average_output = Contrast().contrast_based_edge_detection(image)
    
    contrast_edges_filename = "contrast_edges.png"
    edge_output_filename = "contrast_edge_output.png"
    average_output_filename = "contrast_average_output.png"
    
    contrast_edges_path = os.path.join(TEMP_IMAGE_DIR, contrast_edges_filename)
    edge_output_path = os.path.join(TEMP_IMAGE_DIR, edge_output_filename)
    average_output_path = os.path.join(TEMP_IMAGE_DIR, average_output_filename)
    
    contrast_edges.save(contrast_edges_path)
    edge_output.save(edge_output_path)
    average_output.save(average_output_path)
    
    return [contrast_edges_filename, edge_output_filename, average_output_filename]


def varianceAlgo(image_path):
    image = Image.open(image_path)
    image = FeatureFunction.convert_image_color(image)
    img = VarianceAndRange().variance_operator(image)
    filename = "variance.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename

def rangeAlgo(image_path):
    image = Image.open(image_path)
    image = FeatureFunction.convert_image_color(image)
    img = VarianceAndRange().range_operator(image)
    filename = "range.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename

def highPassAlgo(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = Frequencies().high_pass(image)
    filename = "high_pass.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename

def lowPassAlgo(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = Frequencies().low_pass(image)
    filename = "low_pass.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename


def medianFilterAlgo(image_path):
    image = Image.open(image_path)
    image = FeatureFunction.convert_image_color(image)
    img = Frequencies().median_filter(image)
    filename = "median_filter.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename


def addAlgo(image_path):
    image = Image.open(image_path)
    img = ImageOperation().add_operation(image)
    filename = "add_operation.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename

def subtractAlgo(image_path):
    image = Image.open(image_path)
    img = ImageOperation().subtract_operation(image)
    filename = "subtract_operation.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename

def invertAlgo(image_path):
    image = Image.open(image_path)
    img = ImageOperation().invert_operation(image)
    filename = "invert_operation.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename

def manualSegmentationAlgo(image_path):
    image = Image.open(image_path)
    img = Segmentaion().manual_segmentation(image, 100 , 200)
    filename = "manual_segmentation.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename


def histogramPeakAlgo(image_path):
    image = Image.open(image_path)
    img = Segmentaion().histogram_peak_threshold_segmentation(image)
    filename = "histogram_peak.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename



def histogramValleyAlgo(image_path):
    image = Image.open(image_path)
    img = Segmentaion().histogram_valley_threshold_segmentation(image)
    filename = "histogram_valley.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename

def adaptivehHistogramAlgo(image_path):
    image = Image.open(image_path)
    img = Segmentaion().adaptive_histogram_threshold_segmentation(image)
    filename = "adaptive_histogram.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, filename)
    img.save(img_path)
    return filename


# Map buttons to functions
functions = {
    'button1': halftonedAlg,
    'button2': histogramAlg,
    'button3': sobelAlg,
    'button4': prewittAlg,
    'button5': kirschAlg,
    'button6': homogeneityAlg,
    'button7': differenceAlg,
    'button8': differenceGaussianAlg,
    'button9': contrastAlgo,
    'button10': varianceAlgo,
    'button11': rangeAlgo,
    'button12': highPassAlgo,
    'button13': lowPassAlgo,
    'button14': medianFilterAlgo,
    'button15': addAlgo,
    'button16': subtractAlgo,
    'button17': invertAlgo,
    'button18': manualSegmentationAlgo,
    'button19': histogramPeakAlgo,
    'button20': histogramValleyAlgo,
    'button21': adaptivehHistogramAlgo,
}


@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_filename = None
    processed_image_filenames = [] 

    if request.method == "POST":
        # Handle file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                uploaded_image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(uploaded_image_path)
                uploaded_filename = filename

        # Handle processing buttons
        elif "button_id" in request.form:
            button_id = request.form["button_id"]
            uploaded_filename = request.form.get("uploaded_image_filename")

            if uploaded_filename:
                uploaded_image_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_filename)
                if os.path.exists(uploaded_image_path):
                    action = functions.get(button_id)

                    if action:
                        
                        if button_id == "button8":  
                            processed_image_filenames = action(uploaded_image_path)  

                        elif button_id == "button9":  
                            processed_image_filenames = action(uploaded_image_path)

                        else:
                            processed_image_filenames = [action(uploaded_image_path)]

                        # elif button_id == "button9":  # Contrast
                        #         processed_image_filenames = [
                        #         action(uploaded_image_path),  # First image
                        #         action(uploaded_image_path),  # Second image (modify logic if needed)
                        #         action(uploaded_image_path),  # Third image (modify logic if needed)
                        #     ]
                        # else:
                        #     # For other buttons, process a single image
                        #     processed_image_filenames = [action(uploaded_image_path)]
                else:
                    return render_template("index.html", error="Uploaded image not found.")

    return render_template(
        "index.html",
        uploaded_filename=uploaded_filename,
        processed_image_filenames=processed_image_filenames,
    )





@app.route("/uploaded_images/<filename>")
def uploaded_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/processed_images/<filename>")
def processed_image(filename):
    return send_from_directory(app.config["TEMP_IMAGE_DIR"], filename)

if __name__ == "__main__":
    from waitress import serve
    print("Server running on http://127.0.0.1:8000")
    serve(app, host="0.0.0.0", port=8000)
    # app.run(debug=True)