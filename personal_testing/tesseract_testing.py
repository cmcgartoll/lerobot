import cv2
import pytesseract
import numpy as np
from pytesseract import Output

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Load the image
img_source = cv2.imread('outputs/yolo_training_images/camera_03_frame_000000.png')

def get_grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    """Apply Otsu's thresholding."""
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def adaptive_threshold(image):
    """Apply adaptive Gaussian thresholding."""
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def canny(image):
    """Apply Canny edge detection."""
    return cv2.Canny(image, 30, 150)

def resize(image):
    """Resize the image for better resolution."""
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

def denoise(image):
    """Denoise the image using Non-Local Means."""
    return cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=7, searchWindowSize=21)

# Load the image
img_source = cv2.imread('outputs/yolo_training_images/camera_03_frame_000000.png')

# Apply preprocessing
gray = get_grayscale(img_source)        # Convert to grayscale
gray_denoised = denoise(gray)           # Denoise the grayscale image
scaled = resize(gray_denoised)          # Resize the denoised grayscale image
thresh = thresholding(scaled)           # Apply Otsu's thresholding
adaptive = adaptive_threshold(scaled)   # Apply adaptive thresholding
canny_img = canny(scaled)               # Apply Canny edge detection
custom_config = r'-l eng --oem 3 --psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ" '
        

# Test all preprocessing variations
for i, img in enumerate([img_source, gray, thresh, adaptive, canny_img, scaled]):
    d = pytesseract.image_to_data(img, config=custom_config, lang='eng', output_type=Output.DICT)
    print(d['conf'])
    n_boxes = len(d['text'])

    # Convert grayscale images to RGB for visualization
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for j in range(n_boxes):
        if int(d['conf'][j]) > 60:  # Confidence threshold
            (text, x, y, w, h) = (d['text'][j], d['left'][j], d['top'][j], d['width'][j], d['height'][j])
            if text.strip():
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img = cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save the processed image
    cv2.imwrite(f"processed_output_{i}.png", img)
    print(f"Processed image saved as processed_output_{i}.png")
