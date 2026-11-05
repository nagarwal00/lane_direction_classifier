import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import overlay_verdict_and_curve

def to_binary_mask(
    image: np.ndarray,
    s_thresh: tuple = (100, 255),
    l_thresh: tuple = (180, 255),
    sobel_thresh: tuple = (20, 255)
) -> np.ndarray:
    """
    Convert BGR image to binary mask isolating white/yellow lane markings and high-contrast vertical edges.
    Combines:
    1. HLS S-channel thresholding (color sensitivity for yellow/white)
    2. HLS L-channel thresholding (brightness for white lane lines)
    3. Sobel-X gradient thresholding (vertical edge detection)
    """
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Color Thresholding (S & L channels)
    s_binary = np.zeros_like(s_channel, dtype=np.uint8)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    l_binary = np.zeros_like(l_channel, dtype=np.uint8)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    # 2. Sobel X Gradient Thresholding
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    max_sobel = np.max(abs_sobelx)
    if max_sobel > 0:
        scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel)
    else:
        scaled_sobel = np.uint8(abs_sobelx)
        
    sx_binary = np.zeros_like(scaled_sobel, dtype=np.uint8)
    sx_binary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
    
    # Combined binary mask: (S & L) | Sobel-X
    combined = np.zeros_like(gray, dtype=np.uint8)
    combined[((s_binary == 1) | (l_binary == 1)) | (sx_binary == 1)] = 255
    return combined

def apply_roi_mask(binary_image: np.ndarray) -> np.ndarray:
    """
    Applies a trapezoidal Region of Interest (ROI) mask covering the road region ahead.
    Zeros out sky, trees, bonnet, and roadside clutter.
    """
    height, width = binary_image.shape[:2]
    mask = np.zeros_like(binary_image)
    
    # Trapezoid vertices customized for 1280x720 TuSimple camera perspective
    bottom_left = (int(width * 0.05), height)
    top_left = (int(width * 0.40), int(height * 0.45))
    top_right = (int(width * 0.60), int(height * 0.45))
    bottom_right = (int(width * 0.95), height)
    
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    
    masked = cv2.bitwise_and(binary_image, mask)
    return masked

def detect_lane_pixels(
    masked_binary: np.ndarray,
    nwindows: int = 9,
    margin: int = 80,
    minpix: int = 30
):
    """
    Sliding window technique using histogram peaks on the lower half of the masked binary image.
    Returns pixel coordinates (y, x_left, x_right) belonging to left and right lane lines.
    """
    # Sum histogram of lower half
    bottom_half = masked_binary[masked_binary.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    window_height = int(masked_binary.shape[0] // nwindows)
    
    # Identify non-zero pixel positions in the binary mask
    nonzero = masked_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    out_img = None  # for visual debugging if needed
    
    for window in range(nwindows):
        win_y_low = masked_binary.shape[0] - (window + 1) * window_height
        win_y_high = masked_binary.shape[0] - window * window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify pixels within left and right windows
        good_left_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        
        good_right_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        ).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # Recenter window on mean pixel position if minpix found
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            
    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else np.array([])
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else np.array([])
    
    leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
    lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
    rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
    righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
    
    return lefty, leftx, righty, rightx

def fit_polynomial(pixel_ys: np.ndarray, pixel_xs: np.ndarray):
    """
    Fits a 2nd-degree polynomial x(y) = A*y^2 + B*y + C to detected lane pixels.
    Returns (A, B, C) or None if fewer than 3 valid points.
    """
    if pixel_ys is None or pixel_xs is None or len(pixel_ys) < 3:
        return None
    try:
        poly = np.polyfit(pixel_ys, pixel_xs, 2)
        return poly
    except Exception:
        return None

def process_frame_cv(image: np.ndarray):
    """
    Full CV pipeline for a single image/frame.
    Returns:
    - binary_mask (uint8 0 or 255)
    - masked_roi (uint8 0 or 255)
    - fit_left (A, B, C tuple or None)
    - fit_right (A, B, C tuple or None)
    - fit_center (A, B, C tuple or None)
    - pixel_data dict containing detected lane pixels
    """
    binary_mask = to_binary_mask(image)
    masked_roi = apply_roi_mask(binary_mask)
    
    lefty, leftx, righty, rightx = detect_lane_pixels(masked_roi)
    
    fit_left = fit_polynomial(lefty, leftx)
    fit_right = fit_polynomial(righty, rightx)
    
    # Compute center fit by combining left and right detected pixels or averaging fits
    fit_center = None
    all_y = []
    all_xc = []
    
    # Generate points across ROI y range
    ploty = np.linspace(int(image.shape[0] * 0.45), image.shape[0] - 1, 50)
    
    if fit_left is not None and fit_right is not None:
        left_fitx = fit_left[0] * (ploty ** 2) + fit_left[1] * ploty + fit_left[2]
        right_fitx = fit_right[0] * (ploty ** 2) + fit_right[1] * ploty + fit_right[2]
        center_fitx = (left_fitx + right_fitx) / 2.0
        fit_center = fit_polynomial(ploty, center_fitx)
    elif fit_left is not None:
        left_fitx = fit_left[0] * (ploty ** 2) + fit_left[1] * ploty + fit_left[2]
        center_fitx = left_fitx + 190.0  # estimate offset
        fit_center = fit_polynomial(ploty, center_fitx)
    elif fit_right is not None:
        right_fitx = fit_right[0] * (ploty ** 2) + fit_right[1] * ploty + fit_right[2]
        center_fitx = right_fitx - 190.0  # estimate offset
        fit_center = fit_polynomial(ploty, center_fitx)
        
    pixel_data = {
        "lefty": lefty, "leftx": leftx,
        "righty": righty, "rightx": rightx,
        "ploty": ploty
    }
    
    return binary_mask, masked_roi, fit_left, fit_right, fit_center, pixel_data

if __name__ == "__main__":
    import random
    import glob
    
    images = glob.glob("data/images/*.jpg")
    print(f"CV Pipeline Smoke Test: Found {len(images)} images in data/images/")
    assert len(images) > 0, "No images found for smoke testing!"
    
    sample_50 = random.sample(images, min(50, len(images)))
    print(f"Testing CV pipeline across {len(sample_50)} random sample images...")
    
    success_count = 0
    for img_path in sample_50:
        img = cv2.imread(img_path)
        if img is None:
            continue
        bin_mask, masked_roi, f_left, f_right, f_center, px_data = process_frame_cv(img)
        assert bin_mask is not None and bin_mask.shape == img.shape[:2]
        assert masked_roi is not None and masked_roi.shape == img.shape[:2]
        success_count += 1
        
    print(f"CV Pipeline Smoke Test SUCCESS: Processed {success_count}/{len(sample_50)} images without crash!")
