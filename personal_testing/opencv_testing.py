import cv2
import pytesseract
import numpy as np
from pytesseract import Output

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Load the image
img = cv2.imread('outputs/yolo_training_images/camera_03_frame_000005.png')

def get_grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    """Apply Otsu's thresholding."""
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def canny(image):
    """Apply Canny edge detection."""
    return cv2.Canny(image, 100, 200)

def resize(image):
    """Resize the image for better resolution."""
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

def denoise(image):
    """Denoise the image using Non-Local Means."""
    return cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=7, searchWindowSize=21)

# Apply preprocessing
gray = get_grayscale(img)        # Convert to grayscale
gray_denoised = denoise(gray)           # Denoise the grayscale image
scaled = resize(gray_denoised)          # Resize the denoised grayscale image          # Resize the denoised grayscale image
thresh = thresholding(scaled) 
canny_img = canny(scaled)
cv2.imshow("canny", canny_img)
cv2.waitKey(-1)
items = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = items[0] if len(items) == 2 else items[1]

img_copy = resize(img).copy()
detected = ""
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    ratio = h/w
    area = w*h
    if ratio > 0.8 and ratio < 1.2 and 12000 < area < 30000 and x <= 1150:
        print(f"x: {x}, y: {y}, w: {w}, h: {h}")
        # Get the rotated rectangle
        # Get both rectangles
        min_rect = cv2.minAreaRect(c)  # Returns ((x,y), (width,height), angle)
        # Draw the min area rectangle
        straight_rect = cv2.boundingRect(c)  # Returns (x,y,w,h)

        # The straight rectangle has angle 0 or 90
        # minAreaRect angle is between -90 and 0 degrees
        # Normalize the angle to rotate to nearest 0 or 90 degrees
        # angle = 90 + angle

        # Extract ROI using minAreaRect
        box = cv2.boxPoints(min_rect)
        box = np.intp(box)
        x_min = int(min(box[:, 0]))
        y_min = int(min(box[:, 1]))
        x_max = int(max(box[:, 0]))
        y_max = int(max(box[:, 1]))

        cv2.drawContours(img_copy, [box], 0, (0,255,0), 2)
        cv2.imshow("Min Area Rectangle", img_copy)
        cv2.waitKey(-1)

        angle = min_rect[2]
        print(f"angle: {angle}")
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
            # Rotate box points to match desired orientation
            box = np.roll(box, 1, axis=0)
        print(f"angle after: {angle}")
        # Extract region
        roi = thresh[y_min:y_max, x_min:x_max]

        # Rotate ROI
        center = (roi.shape[1] // 2, roi.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_roi = cv2.warpAffine(roi, matrix, (roi.shape[1], roi.shape[0]))
        # print(rotated_roi.shape)

        # cv2.imshow("Rotated to align with axes", rotated_roi)
        # cv2.waitKey(-1)

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
        # cv2.imshow("110x110 tile", final_tile)
        # cv2.waitKey(-1)
        
        # cv2.imwrite(f'outputs/tests/rotated_segment_{x}_{y}.png', rotated_roi)
        # Try all 4 rotations (0, 90, 180, 270 degrees) until valid letter found
        valid_letter = False
        best_angle = -1
        custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ" '
        ROTATIONS = 4
        for rotation in range(ROTATIONS):
            if rotation == 0:
                rotated_tile = final_tile
            elif rotation == 1:
                rotated_tile = cv2.rotate(final_tile, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 2:
                rotated_tile = cv2.rotate(final_tile, cv2.ROTATE_180)
            else:
                rotated_tile = cv2.rotate(final_tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Try OCR on this rotation
            letter = pytesseract.image_to_string(rotated_tile, config=custom_config).strip()
            print(letter)
            if len(letter) == 1 and letter.isalpha():
                valid_letter = True
                detected = detected + letter.replace('\n', '')
                best_angle = rotation * 90
                cv2.imshow("letter", rotated_tile)
                cv2.waitKey(-1)
                print(f"best_angle: {best_angle}")
                final_tile = rotated_tile  # Keep the correctly oriented tile
                break
                
        # Draw the min area rectangle and highlight top edge
        cv2.drawContours(img_copy, [box], 0, (0,255,0), 2)
        
        # Get the points for top and bottom halves based on rotation
        match best_angle:
            case 0:  # Original top is still top
                top_half = np.array([box[1], box[2], box[3]], dtype=np.int32)
                bottom_half = np.array([box[3], box[0], box[1]], dtype=np.int32)
            case 90:  # Right edge became top
                top_half = np.array([box[2], box[3], box[0]], dtype=np.int32)
                bottom_half = np.array([box[0], box[1], box[2]], dtype=np.int32)
            case 180:  # Bottom edge became top
                top_half = np.array([box[3], box[0], box[1]], dtype=np.int32)
                bottom_half = np.array([box[1], box[2], box[3]], dtype=np.int32)
            case 270:  # Left edge became top
                top_half = np.array([box[0], box[1], box[2]], dtype=np.int32)
                bottom_half = np.array([box[2], box[3], box[0]], dtype=np.int32)
            case _:
                print(f"Did not find a letter")
        
        # Fill the top half in blue and bottom half in green
        # Create overlay image for transparency
        if best_angle != -1:
            overlay = img_copy.copy()
            cv2.fillPoly(overlay, [top_half], (0, 0, 255))  # Red
            cv2.fillPoly(overlay, [bottom_half], (0, 255, 0))  # Green
            # Apply transparency
            cv2.addWeighted(overlay, 0.5, img_copy, 0.5, 0, img_copy)
            cv2.imshow("Letters with orientation", img_copy)
            cv2.waitKey(-1)

        # custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ" '
        # c = pytesseract.image_to_string(final_tile, config=custom_config)
        # print('contour: ', c)
        # detected = detected + c.replace('\n', '')

print("detected: ", detected)
