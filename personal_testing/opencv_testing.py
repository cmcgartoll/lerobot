import cv2
import pytesseract
import numpy as np
from pytesseract import Output

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Load the image
img = cv2.imread('outputs/yolo_training_images/camera_03_frame_000013.png')

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
    return cv2.Canny(image, 30, 50)

def resize(image):
    """Resize the image for better resolution."""
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

def denoise(image):
    """Denoise the image using Non-Local Means."""
    return cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=7, searchWindowSize=21)

# Apply preprocessing
gray = get_grayscale(img)        # Convert to grayscale
gray_denoised = denoise(gray)           # Denoise the grayscale image
scaled = resize(gray_denoised)          # Resize the denoised grayscale image
thresh = thresholding(scaled) 
adaptive = adaptive_threshold(scaled)   # Apply adaptive thresholding
canny_img = canny(scaled)
cv2.imshow("canny", canny_img)
cv2.waitKey(-1)
items = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = items[0] if len(items) == 2 else items[1]

img_copy = (scaled).copy()
img_contour = scaled.copy()
boxes = []
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    x, y, w, h = cv2.boundingRect(contours[i])
    ratio = h/w
    area = cv2.contourArea(contours[i])
    base = np.ones(thresh.shape, dtype=np.uint8)
    if ratio > 0.8 and 13000 < w*h < 30000 and x > 1150:
    # if 80 < area < 500:
        # print(area, w*h)
        out = cv2.drawContours(img_contour, contours, i, (0, 0, 255), 2)
        cv2.drawContours(img_copy, contours, i, (0, 0, 255), 2)

        x, y, w, h = cv2.boundingRect(contours[i])

        # Draw the rectangle on the original image copy
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)  # Get the four vertices of the rotated rectangle
        box = np.intp(box)  # Convert points to integers

        # Draw the rotated rectangle on the image
        cv2.drawContours(img_copy, [box], 0, (255, 0, 0), 2)

cv2.imshow("Original Image with Bounding Rectangles", img_copy)
cv2.waitKey(0)
detected = ""
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    
    ratio = h/w
    area = cv2.contourArea(c)
    base = np.ones(thresh.shape, dtype=np.uint8)
    if ratio > 0.8 and 13000 < w*h < 30000 and x > 1150:
        base[y:y+h, x:x+w] = thresh[y:y+h, x:x+w]
        segment = cv2.bitwise_not(base)
        
        # Get the rotated rectangle
        # Get both rectangles
        min_rect = cv2.minAreaRect(c)  # Returns ((x,y), (width,height), angle)
        straight_rect = cv2.boundingRect(c)  # Returns (x,y,w,h)

        # The straight rectangle has angle 0 or 90
        # minAreaRect angle is between -90 and 0 degrees
        angle = min_rect[2]

        # Normalize the angle to rotate to nearest 0 or 90 degrees
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        # Extract ROI using minAreaRect
        box = cv2.boxPoints(min_rect)
        box = np.intp(box)
        x_min = int(min(box[:, 0]))
        y_min = int(min(box[:, 1]))
        x_max = int(max(box[:, 0]))
        y_max = int(max(box[:, 1]))

        # Extract region
        roi = thresh[y_min:y_max, x_min:x_max]

        # Rotate ROI
        center = (roi.shape[1] // 2, roi.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_roi = cv2.warpAffine(roi, matrix, (roi.shape[1], roi.shape[0]))
        print(rotated_roi.shape)

        cv2.imshow("Rotated to align with axes", rotated_roi)
        cv2.waitKey(-1)

        h, w = rotated_roi.shape
        center_y, center_x = h//2, w//2
        crop_size = 110

        # Calculate crop coordinates (centered)
        x_start = center_x - crop_size//2
        y_start = center_y - crop_size//2

        # Crop exactly crop_size x crop_size from center
        final_tile = rotated_roi[y_start:y_start+crop_size, x_start:x_start+crop_size]

        # Verify size
        print(f"Final tile shape: {final_tile.shape}")
        cv2.imshow("124x124 tile", final_tile)
        cv2.waitKey(-1)
        
        cv2.imwrite(f'outputs/tests/rotated_segment_{x}_{y}.png', rotated_roi)

        custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ" '
        c = pytesseract.image_to_string(final_tile, config=custom_config)
        print('contour: ', c)
        detected = detected + c.replace('\n', '')
        # cv2.imshow("segment", segment)
        # cv2.waitKey(-1)
        # input('enter')

print("detected: ", detected)
