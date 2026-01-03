#!/usr/bin/env python3
import cv2
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# CONFIGURATION
VIDEO_PATH = "data/easy/game_easy.mp4"
OUTPUT_DIR = "output_video"
SAVE_VIDEO = True

@dataclass
class DetectionParams:
    """
    Parameters for card detection
    """
    # Edge/Gradient settings
    gradient_threshold: int = 30
    canny_low: int = 50
    canny_high: int = 150
    
    # Morphological closing
    close_kernel_size: int = 7
    close_iterations: int = 4
    
    # Geometric Filters
    min_area: int = 8500
    max_area: int = 18000
    min_aspect: float = 0.6
    max_aspect: float = 3.0
    min_extent: float = 0.5
    max_extent: float = 0.95
    solidity_threshold: float = 0.8
    
    # Card dimension reference
    card_width: int = 110
    card_height: int = 150

class CardDetector:
    def __init__(self, params=None):
        self.params = params if params else DetectionParams()
    
    def split_merged_cards(self, single_card_mask, rect):
        """
        Split a mask containing AT MOST 2 cards (1x2 or 2x1).
        Rejects anything larger than 2 cards.
        """
        width, height = rect[1]
        angle = rect[2]
        
        # Test all possible configurations for 1 or 2 cards
        # Portrait cards: 100w × 140h
        # Landscape cards: 140w × 100h
        
        configs = []
        
        # Single portrait card
        error_1p = abs(width - self.params.card_width) + abs(height - self.params.card_height)
        configs.append((1, 1, error_1p, 'portrait'))
        
        # Single landscape card
        error_1l = abs(width - self.params.card_height) + abs(height - self.params.card_width)
        configs.append((1, 1, error_1l, 'landscape'))
        
        # Two cards horizontally (2×1 portrait)
        error_2h_p = abs(width - 2 * self.params.card_width) + abs(height - self.params.card_height)
        configs.append((2, 1, error_2h_p, 'portrait'))
        
        # Two cards horizontally (2×1 landscape)
        error_2h_l = abs(width - 2 * self.params.card_height) + abs(height - self.params.card_width)
        configs.append((2, 1, error_2h_l, 'landscape'))
        
        # Two cards vertically (1×2 portrait)
        error_2v_p = abs(width - self.params.card_width) + abs(height - 2 * self.params.card_height)
        configs.append((1, 2, error_2v_p, 'portrait'))
        
        # Two cards vertically (1×2 landscape)
        error_2v_l = abs(width - self.params.card_height) + abs(height - 2 * self.params.card_width)
        configs.append((1, 2, error_2v_l, 'landscape'))
        
        # Find best fit
        configs.sort(key=lambda x: x[2])  # Sort by error
        cols, rows, best_error, orientation = configs[0]
        
        # If error is too large (>50 pixels total), this is probably noise or >2 cards
        # Reject it
        if best_error > 50:
            return []  # Reject this region entirely
        
        # Single card - return as-is
        if cols == 1 and rows == 1:
            return [(rect, width, height)]
        
        # Rotate mask to be axis-aligned
        center = rect[0]
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        h, w = single_card_mask.shape
        rotated_mask = cv2.warpAffine(single_card_mask, M, (w, h))
        
        # Find bounding box
        coords = cv2.findNonZero(rotated_mask)
        if coords is None:
            return [(rect, width, height)]
        
        x, y, w_box, h_box = cv2.boundingRect(coords)
        
        # Safety check: ensure we have valid dimensions
        if w_box == 0 or h_box == 0 or cols == 0 or rows == 0:
            return [(rect, width, height)]
        
        # Calculate cell dimensions
        card_w = w_box / cols
        card_h = h_box / rows
        
        split_rects = []
        
        # Check each grid cell
        for row in range(rows):
            for col in range(cols):
                cell_x = x + col * card_w
                cell_y = y + row * card_h
                
                cell_x1, cell_y1 = int(cell_x), int(cell_y)
                cell_x2, cell_y2 = int(cell_x + card_w), int(cell_y + card_h)
                
                # Bounds check
                cell_x1 = max(0, cell_x1)
                cell_y1 = max(0, cell_y1)
                cell_x2 = min(rotated_mask.shape[1], cell_x2)
                cell_y2 = min(rotated_mask.shape[0], cell_y2)
                
                cell_roi = rotated_mask[cell_y1:cell_y2, cell_x1:cell_x2]
                
                # Check if cell contains a card (>30% filled)
                if cell_roi.size > 0:
                    fill_ratio = np.count_nonzero(cell_roi) / cell_roi.size
                    
                    if fill_ratio > 0.3:
                        # Calculate center in rotated space
                        cell_center_x = cell_x + card_w / 2
                        cell_center_y = cell_y + card_h / 2
                        
                        # Rotate back to original orientation
                        inv_M = cv2.getRotationMatrix2D(center, -angle, 1.0)
                        cell_center = np.array([[[cell_center_x, cell_center_y]]], dtype=np.float32)
                        orig_center = cv2.transform(cell_center, inv_M)[0][0]
                        
                        card_rect = (
                            (orig_center[0], orig_center[1]),
                            (card_w, card_h),
                            angle  # Keep original angle, no adjustment needed
                        )
                        split_rects.append((card_rect, card_w, card_h))
        
        return split_rects if split_rects else [(rect, width, height)]
    
    def detect_cards(self, frame):
        """Detect all cards in a frame and return their bounding boxes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Morphological Gradient
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel_grad)
        
        # Threshold gradient
        _, gradient_binary = cv2.threshold(
            gradient, 
            self.params.gradient_threshold, 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Canny edges
        edges = cv2.Canny(
            gradient_binary, 
            self.params.canny_low, 
            self.params.canny_high, 
            apertureSize=3
        )
        
        # Connect edges and fill contours
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        connected_edges = cv2.dilate(edges, kernel_connect, iterations=2)
        
        contours, _ = cv2.findContours(
            connected_edges, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Fill contours
        filled_mask = np.zeros_like(gray)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                cv2.drawContours(filled_mask, [contour], -1, (255), cv2.FILLED)
        
        # Watershed segmentation
        kernel_clean = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_clean, iterations=2)
        sure_bg = cv2.dilate(opening, kernel_clean, iterations=3)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(frame, markers)
        
        # Extract and split cards
        valid_boxes = []
        unique_markers = np.unique(markers)
        
        for label in unique_markers:
            if label <= 1:
                continue
            
            single_card_mask = np.zeros(gray.shape, dtype="uint8")
            single_card_mask[markers == label] = 255
            
            cnts, _ = cv2.findContours(single_card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(cnts) > 0:
                c = cnts[0]
                area = cv2.contourArea(c)
                rect = cv2.minAreaRect(c)
                width, height = rect[1]
                
                if width == 0 or height == 0:
                    continue
                
                if area < (self.params.min_area * 0.5):
                    continue
                
                # Split merged cards
                split_cards = self.split_merged_cards(single_card_mask, rect)
                
                for card_data in split_cards:
                    card_rect, _, _ = card_data
                    box = cv2.boxPoints(card_rect)
                    box = np.int32(box)
                    valid_boxes.append(box)
        
        return valid_boxes

def main():
    video_path = Path(VIDEO_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # Video Properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.2f} fps")
    print(f"Total frames: {total_frames}")

    # Initialize Video Writer
    if SAVE_VIDEO:
        out_name = output_dir / f"detected_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_name), fourcc, fps, (width, height))
        print(f"Output will be saved to: {out_name}")

    detector = CardDetector()
    
    print(f"\nProcessing frames...")
    start_time = time.time()

    frame_count = 0
    card_id = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect cards
        boxes = detector.detect_cards(frame)

        # Draw results
        for box in boxes:
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            
            # Calculate center and dimensions
            rect = cv2.minAreaRect(box)
            center_x, center_y = int(rect[0][0]), int(rect[0][1])
            w, h = rect[1]
            if w > h:
                w, h = h, w
            
            # Draw card ID and dimensions
            cv2.putText(frame, f"#{card_id}", (center_x, center_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"W:{int(w)} H:{int(h)}", (center_x, center_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
            card_id += 1

        if SAVE_VIDEO:
            writer.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Frame {frame_count}/{total_frames} | Speed: {frame_count/elapsed:.2f} fps | Cards: {len(boxes)}")

        # Reset card_id for next frame
        card_id = 1

    cap.release()
    if SAVE_VIDEO:
        writer.release()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Finished!")
    print(f"Processed {frame_count} frames in {elapsed:.2f}s")
    print(f"Average speed: {frame_count/elapsed:.2f} fps")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main()