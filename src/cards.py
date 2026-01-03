#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_PATH = "data/easy/game_easy.mp4"  # Change this to your video path
OUTPUT_DIR = "output_minimal_gradient"

@dataclass
class DetectionParams:
    """
    Exact parameters from board2.py used for Gradient/Card detection
    """
    # Edge/Gradient settings
    gradient_threshold: int = 30   # Threshold for the morphological gradient
    canny_low: int = 50            # Canny edge detection lower bound
    canny_high: int = 150          # Canny edge detection upper bound
    
    # Morphological closing (connecting gaps)
    close_kernel_size: int = 7
    close_iterations: int = 4
    
    # Geometric Filters (filtering noise vs actual cards)
    min_area: int = 8500
    max_area: int = 18000
    min_aspect: float = 0.6
    max_aspect: float = 3.0
    min_extent: float = 0.5
    max_extent: float = 0.95
    solidity_threshold: float = 0.8
    
    # Card dimension reference (for splitting merged cards)
    card_width: int = 90
    card_height: int = 130

def split_merged_cards(single_card_mask, rect, params):
    """
    Split a mask that contains multiple cards into individual card regions.
    Returns a list of tuples: (card_rect, original_width, original_height)
    """
    width, height = rect[1]
    angle = rect[2]
    original_angle = angle
    
    # Normalize dimensions (width should be smaller, height larger for portrait cards)
    if width > height:
        width, height = height, width
        angle += 90
    
    # Determine how many cards are in this region
    # Allow 15% tolerance for card size variations
    cols = round(width / params.card_width)
    rows = round(height / params.card_height)
    
    # If it's just a single card, return with original dimensions
    if cols == 1 and rows == 1:
        # Return original rect dimensions (before normalization)
        return [(rect, rect[1][0], rect[1][1])]
    
    # Get the rotation matrix to align the card(s) horizontally
    center = rect[0]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the mask to be axis-aligned
    h, w = single_card_mask.shape
    rotated_mask = cv2.warpAffine(single_card_mask, M, (w, h))
    
    # Find the bounding box of the rotated region
    coords = cv2.findNonZero(rotated_mask)
    if coords is None:
        return [(rect, rect[1][0], rect[1][1])]
    
    x, y, w_box, h_box = cv2.boundingRect(coords)
    if w_box == 0 or h_box == 0 or cols == 0 or rows == 0:
            return [(rect, rect[1][0], rect[1][1])]
    # Calculate individual card dimensions within the grid
    card_w = w_box / cols
    card_h = h_box / rows
    
    # Split into grid cells and check which ones contain cards
    split_rects = []
    
    for row in range(rows):
        for col in range(cols):
            # Define the cell boundaries
            cell_x = x + col * card_w
            cell_y = y + row * card_h
            
            # Extract this cell from the rotated mask
            cell_x1, cell_y1 = int(cell_x), int(cell_y)
            cell_x2, cell_y2 = int(cell_x + card_w), int(cell_y + card_h)
            
            # Ensure bounds are within image
            cell_x1 = max(0, cell_x1)
            cell_y1 = max(0, cell_y1)
            cell_x2 = min(rotated_mask.shape[1], cell_x2)
            cell_y2 = min(rotated_mask.shape[0], cell_y2)
            
            cell_roi = rotated_mask[cell_y1:cell_y2, cell_x1:cell_x2]
            
            # Check if this cell actually contains a card
            # A cell is considered to have a card if at least 30% of it is filled
            if cell_roi.size > 0:
                fill_ratio = np.count_nonzero(cell_roi) / cell_roi.size
                
                if fill_ratio > 0.3:
                    # This cell has a card! Calculate its center in rotated space
                    cell_center_x = cell_x + card_w / 2
                    cell_center_y = cell_y + card_h / 2
                    
                    # Rotate this center back to original orientation
                    inv_M = cv2.getRotationMatrix2D(center, -angle, 1.0)
                    cell_center = np.array([[[cell_center_x, cell_center_y]]], dtype=np.float32)
                    orig_center = cv2.transform(cell_center, inv_M)[0][0]
                    
                    # Create a rotated rect for this individual card
                    # Store the rect along with the ORIGINAL width and height (before rotation)
                    card_rect = (
                        (orig_center[0], orig_center[1]),  # center
                        (card_w, card_h),  # size (normalized)
                        angle  # angle (normalized)
                    )
                    split_rects.append((card_rect, card_w, card_h))
    
    return split_rects if split_rects else [(rect, rect[1][0], rect[1][1])]

def process_gradient_detection():
    # 1. Setup
    video_path = Path(VIDEO_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = DetectionParams()
    
    print(f"Processing: {video_path}")
    print(f"Outputting to: {output_dir}")

    # 2. Load Frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Save original
    cv2.imwrite(str(output_dir / "0_original.png"), frame)

    # =========================================================
    # STEP 1: PRE-PROCESSING & GRADIENT
    # =========================================================
    print("Step 1: Calculating Morphological Gradient...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate Morphological Gradient (Difference between dilation and erosion)
    # This highlights areas of high contrast (edges/details on cards)
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel_grad)
    
    # Save visualization
    cv2.imwrite(str(output_dir / "1_gradient_raw.png"), gradient)

    # =========================================================
    # STEP 2: THRESHOLDING
    # =========================================================
    print("Step 2: Thresholding Gradient...")
    # Keep only strong gradient areas
    _, gradient_binary = cv2.threshold(
        gradient, 
        params.gradient_threshold, 
        255, 
        cv2.THRESH_BINARY
    )
    
    cv2.imwrite(str(output_dir / "2_gradient_binary.png"), gradient_binary)

    # =========================================================
    # STEP 3: EDGE DETECTION (CANNY)
    # =========================================================
    print("Step 3: Canny Edge Detection...")
    edges = cv2.Canny(
        gradient_binary, 
        params.canny_low, 
        params.canny_high, 
        apertureSize=3
    )
    
    cv2.imwrite(str(output_dir / "3_canny_edges.png"), edges)

    # =========================================================
    # STEP 4: MORPHOLOGICAL CLOSING
    # =========================================================
    # 1. Light dilation just to connect broken lines in the border
    # We don't try to fill the whole card, just make the border solid
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    connected_edges = cv2.dilate(edges, kernel_connect, iterations=2)
    
    # 2. Find contours of these "hollow" boxes
    contours, hierarchy = cv2.findContours(
        connected_edges, 
        cv2.RETR_EXTERNAL,  # Only outer contours (ignores holes inside)
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 3. Create a mask and explicitly FILL the contours
    filled_mask = np.zeros_like(gray)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000: # Filter tiny noise
            # thickness=cv2.FILLED (-1) colors the inside white
            cv2.drawContours(filled_mask, [contour], -1, (255), cv2.FILLED)
            
    cv2.imwrite(str(output_dir / "4_force_filled.png"), filled_mask)
    
    # =========================================================
    # STEP 5: SEPARATING TOUCHING CARDS (Watershed)
    # =========================================================
    print("Step 5: Separating touching cards...")

    # 1. NOISE REMOVAL
    # Remove tiny dots so they don't become "centers" of cards
    kernel_clean = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_clean, iterations=2)

    # 2. FIND SURE BACKGROUND (The definitely black area)
    # Dilate the object to grab a bit of background
    sure_bg = cv2.dilate(opening, kernel_clean, iterations=3)

    # 3. FIND SURE FOREGROUND (The centers of the cards)
    # Distance Transform: Calculates how far each pixel is from the nearest zero pixel.
    # The "peaks" of this mountain are the centers of our cards.
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # We threshold to keep only the "peaks". 
    # 0.4 means "keep pixels that are at least 40% of the max distance from the edge"
    # Adjust this (0.3 - 0.6) if centers are merging or disappearing.
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 4. FIND UNKNOWN REGION (The border area where cards might touch)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5. MARKERS (Label the centers: 1, 2, 3...)
    # This gives every distinct "center" a unique ID number.
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all backgrounds so sure_bg is not 0, but 1
    markers = markers + 1
    # Mark the unknown region with 0
    markers[unknown == 255] = 0

    # 6. WATERSHED ALGORITHM
    # This grows the markers (1, 2, 3...) into the unknown region
    # until they hit the "walls" defined by the image gradients.
    # It draws the boundary lines with -1.
    markers = cv2.watershed(frame, markers)
    
    # =========================================================
    # EXTRACT BOXES FROM MARKERS AND SPLIT MERGED CARDS
    # =========================================================
    valid_cards = []
    vis_contours = frame.copy()
    
    # Counter for unique card IDs
    card_id = 1
    
    # Loop through unique markers (skip 0=unknown, 1=background)
    unique_markers = np.unique(markers)
    
    for label in unique_markers:
        if label <= 1: continue  # Skip background
        
        # Create a mask for ONLY this specific card ID
        # This isolates one card from the clump!
        single_card_mask = np.zeros(gray.shape, dtype="uint8")
        single_card_mask[markers == label] = 255
        
        # Find contours on this ISOLATED mask
        cnts, _ = cv2.findContours(single_card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            c = cnts[0] # There should be only one main contour per marker
            area = cv2.contourArea(c)
            
            # --- FILTERING LOGIC ---
            # Now we can check if this specific separated chunk looks like a card
            rect = cv2.minAreaRect(c)
            width, height = rect[1]
            
            if width == 0 or height == 0: continue
            
            # Since Watershed might erode the borders slightly, 
            # we might want to be a bit more lenient on min_area
            if area < (params.min_area * 0.5): 
                continue
            
            # Try to split this region into individual cards
            split_cards = split_merged_cards(single_card_mask, rect, params)
            
            # Draw each split card with unique ID
            for card_data in split_cards:
                card_rect, orig_w, orig_h = card_data
                
                box = cv2.boxPoints(card_rect)
                box = np.int32(box)
                valid_cards.append(box)
                cv2.drawContours(vis_contours, [box], 0, (0, 255, 0), 2)
                
                # Display dimensions - always show smaller value as width
                w, h = orig_w, orig_h
                if w > h:
                    w, h = h, w
                dim_text = f"W:{int(w)} H:{int(h)}"
                
                # Draw unique card ID and dimensions
                center_x, center_y = int(card_rect[0][0]), int(card_rect[0][1])
                cv2.putText(vis_contours, f"#{card_id}", (center_x, center_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(vis_contours, dim_text, (center_x, center_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Increment card ID for next card
                card_id += 1

    cv2.imwrite(str(output_dir / "5_final_detections.png"), vis_contours)
    
    # DEBUG: Save the distance transform to see the "peaks"
    cv2.imwrite(str(output_dir / "5b_debug_distance.png"), cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX))
    
    print("-" * 40)
    print(f"Found {len(valid_cards)} potential cards.")
    print(f"Images saved to: {output_dir}")
    print("  1_gradient_raw.png    : Shows high contrast texture")
    print("  2_gradient_binary.png : Thresholded texture")
    print("  3_canny_edges.png     : Edges of the texture")
    print("  4_force_filled.png    : Filled contours")
    print("  5_final_detections.png: Green boxes with W/H dimensions")

if __name__ == "__main__":
    process_gradient_detection()