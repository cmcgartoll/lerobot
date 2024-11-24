import cv2
import pytesseract
import numpy as np
from pytesseract import Output
import argparse

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def get_grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    """Apply Otsu's thresholding."""
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def canny(image):
    """Apply Canny edge detection."""
    return cv2.Canny(image, 100, 200)

def resize(image, scaling_factor):
    """Resize the image for better resolution."""
    return cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

def denoise(image):
    """Denoise the image using Non-Local Means."""
    return cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=7, searchWindowSize=21)

def process_image(img_path, orig_letter, end_location):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image at {img_path}")
        return False
    try:
        processed_img = highlight_letter_and_end_location(img, orig_letter, end_location)
        if processed_img is not None:
            cv2.imwrite(img_path, processed_img)
            return True
        else:
            print(f"Failed to process image {img_path}: highlight_letter_and_end_location returned None")
            return False
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        return False

def highlight_letter_and_end_location(img, orig_letter, end_location):
    scaling_factor = 2
    # Apply preprocessing
    gray = get_grayscale(img)        # Convert to grayscale
    gray_denoised = denoise(gray)           # Denoise the grayscale image
    scaled = resize(gray_denoised, scaling_factor)          # Resize the denoised grayscale image
    thresh = thresholding(scaled) 
    canny_img = canny(scaled)
    # cv2.imshow("canny_img", canny_img)
    # cv2.waitKey(0)
    items = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = items[0] if len(items) == 2 else items[1]
    img_copy = resize(img, scaling_factor).copy()
    letter_found = False
    end_location_found = False
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = h/w
        area = w*h
        
        if ratio > 0.8 and ratio < 1.2 and 1250 < area < 3000 and x > 400 and not letter_found:

            if highlight_letter(c, orig_letter, img_copy, thresh, img):
                letter_found = True
        if ratio > 0.8 and ratio < 1.2 and 1250 < area < 3000 and x <= 400 and not end_location_found:
            if highlight_end_location(c, end_location, img_copy):
                end_location_found = True
    img_copy = cv2.resize(img_copy, (img.shape[1], img.shape[0]))
    return img_copy

def highlight_letter(c, orig_letter, img_copy, thresh, img):
    min_rect = cv2.minAreaRect(c)  # Returns ((x,y), (width,height), angle)
    # Extract ROI using minAreaRect
    box = cv2.boxPoints(min_rect)
    box = np.intp(box)
    x_min = int(min(box[:, 0]))
    y_min = int(min(box[:, 1]))
    x_max = int(max(box[:, 0]))
    y_max = int(max(box[:, 1]))

    angle = min_rect[2]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
        # Rotate box points to match desired orientation
        box = np.roll(box, 1, axis=0)
    # Extract region
    roi = thresh[y_min:y_max, x_min:x_max]

    # Rotate ROI
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_roi = cv2.warpAffine(roi, matrix, (roi.shape[1], roi.shape[0]))
    h, w = rotated_roi.shape[:2]
    center_y, center_x = h//2, w//2
    crop_size = 35

    # Calculate crop coordinates (centered)
    x_start = center_x - crop_size//2
    y_start = center_y - crop_size//2

    # Crop exactly crop_size x crop_size from center
    final_tile = rotated_roi[y_start:y_start+crop_size, x_start:x_start+crop_size]
    # cv2.ims how("final_tile", final_tile)
    # cv2.waitKey(0)

    # Try all 4 rotations (0, 90, 180, 270 degrees) until valid letter found
    best_angle = -1
    custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ" '
    ROTATIONS = 4
    letter_found = ''
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
        if len(letter) == 1 and letter.isalpha():
            best_angle = rotation * 90
            letter_found = letter
            final_tile = rotated_tile  # Keep the correctly oriented tile
            break
    if letter_found != orig_letter:
        return
    # Draw the min area rectangle and highlight top edge
    cv2.drawContours(img_copy, [box], 0, (0,255,0), 5)
    
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
        # Fill polygons directly on img_copy without transparency
        cv2.fillPoly(img_copy, [top_half], (0, 0, 255))  # Red
        cv2.fillPoly(img_copy, [bottom_half], (0, 255, 0))  # Green
        return True
    return False


def highlight_end_location(c, end_location, img_copy):
    min_rect = cv2.minAreaRect(c)  # Returns ((x,y), (width,height), angle)
    _,y = min_rect[0]
    if (end_location == 1 and y < 795 and y >= 715) or \
       (end_location == 2 and y < 715 and y >= 635) or \
       (end_location == 3 and y < 635 and y >= 555) or \
       (end_location == 4 and y < 555 and y >= 475) or \
       (end_location == 5 and y < 475 and y >= 395):
        # Extract ROI using minAreaRect
        box = cv2.boxPoints(min_rect)
        box = np.intp(box)
        # Fill the box with yellow and make outline yellow
        cv2.drawContours(img_copy, [box], 0, (0,255,255), 5)  # Yellow outline
        cv2.fillPoly(img_copy, [box], (0,255,255))  # Yellow fill
    else:
        return False
    return True
    
    
    return img_copy
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Highlight given letters in an image. It will highlight only one letter in the case of multiple of the same letter."
    )
    parser.add_argument(
        "--letter",
        type=str,
        required=True,
        default=None,
        help="Give the letter to highlight.",
    )
    parser.add_argument(
        "--end-location",
        type=int,
        required=True,
        default=None,
        help="Location where letter should be placed (1-5).",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Give the folder to highlight letters in.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Give the number of frames to highlight letters in the folder.",
    )
    args = parser.parse_args()

    for frame in range(args.num_frames):
        print(f"Processing frame {frame:06d}")
        img = cv2.imread(f"{args.folder}/frame_{frame:06d}.png")
        new_img = highlight_letter_and_end_location(img, args.letter, args.end_location)
        if new_img is None:
            print(f"Did not find letter {args.letter}, manually check")
            continue
        cv2.imwrite(f"outputs/transformed_images/{args.folder.split('/')[-1]}_frame_{frame:06d}_{args.letter}_{args.end_location}.png", new_img)

