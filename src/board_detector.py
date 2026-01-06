import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import json


class BoardDetector:
    """Detects Summoner Wars game board using brightness mask and adaptive grid"""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.board_history = []
        self.inner_board = None
        self.grid_cols = 8
        self.grid_rows = 6
        self.calibration_visualized = False
        
    def calibrate(self, frame: np.ndarray, manual_corners: Optional[np.ndarray] = None):
        """Calibrate detector with first frame"""
        if manual_corners is not None:
            self.inner_board = manual_corners
            return manual_corners
        
        inner = self._detect_inner_from_brightness(frame)
        self.inner_board = inner
        return inner
    
    def find_edges_with_white_threshold(self, binary_img, corners, dx=5, threshold_white=0.8):
        """
        Find the x-coordinates where the percentage of white pixels reaches the threshold
        from both left and right edges, and update the corners accordingly.
    
        Args:
            binary_img: Binarized image (white = 255, black = 0)
            corners: Array of 4 corners [[x1,y1], [x1,y2], [x2,y1], [x2,y2]]
            dx: Step size for scanning
            threshold: White pixel percentage threshold (0.0 to 1.0)
    
        Returns:
            Updated corners array with new x-coordinates
        """
        # Extract coordinates
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]

        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))  

        # Scan from left (smaller x)
        x_current = x_min
        x_left = x_min
    
        while x_current + dx <= x_max:
            x1 = x_current
            x2 = x_current + dx
        
            region = binary_img[y_min:y_max, x1:x2]
        
            total_pixels = region.size
            white_pixels = np.sum(region == 255)
            white_percentage = white_pixels / total_pixels if total_pixels > 0 else 0
        
            if white_percentage >= threshold_white:
                x_left = x_current
                break
            
            x_current += dx
        else:
            x_left = x_current

        # Scan from right (bigger x)
        x_current = x_max
        x_right = x_max

        while x_current - dx >= x_min:
            x1 = x_current - dx
            x2 = x_current

            region = binary_img[y_min:y_max, x1:x2]
        
            total_pixels = region.size
            white_pixels = np.sum(region == 255)
            white_percentage = white_pixels / total_pixels if total_pixels > 0 else 0
        
            if white_percentage >= threshold_white:
                x_right = x_current
                break
                
            x_current -= dx
        else:
            x_right = x_current
    
        # Update corners with new x coordinates
        corners_updated = corners.copy()
    
        # Replace all occurrences of x_min with x_left
        corners_updated[corners_updated[:, 0] == x_min, 0] = x_left

        # Replace all occurrences of x_max with x_right
        corners_updated[corners_updated[:, 0] == x_max, 0] = x_right

        return corners_updated


    def refine_both_edges(self, binary_img, corners, dx=5, threshold_refine=1.001):
        """
        Refine both left and right edges by adjusting corners iteratively.
        Stops when the percentage of white pixels decreases for either edge.

        Args:
            binary_img: Binarized image (white = 255, black = 0)
            corners: Array of 4 corners with shape (4, 2)
            dx: Step size for adjustment
            threshod_refine: controls how often and how much refinement will occur

        Returns:
            Updated corners array
        """
        corners_updated = corners.copy()

        # Find the two left corners (smallest x coordinates)
        x_coords = corners_updated[:, 0]
        left_indices = np.argsort(x_coords)[:2]
        right_indices = np.argsort(x_coords)[-2:]

        # Determine upper and lower for left corners
        if corners_updated[left_indices[0], 1] < corners_updated[left_indices[1], 1]:
            upper_left_idx = left_indices[0]
            lower_left_idx = left_indices[1]
        else:
            upper_left_idx = left_indices[1]
            lower_left_idx = left_indices[0]

        # Determine upper and lower for right corners
        if corners_updated[right_indices[0], 1] < corners_updated[right_indices[1], 1]:
            upper_right_idx = right_indices[0]
            lower_right_idx = right_indices[1]
        else:
            upper_right_idx = right_indices[1]
            lower_right_idx = right_indices[0]
    
        def compute_white_percentage(corners_temp):
            """Calculate percentage of white pixels in the quadrilateral."""
            mask = np.zeros_like(binary_img, dtype=np.uint8)
            corners_int = corners_temp.astype(np.int32)
            cv2.fillPoly(mask, [corners_int], 255)

            region = cv2.bitwise_and(binary_img, mask)

            total_pixels = np.sum(mask == 255)
            white_pixels = np.sum(region == 255)

            return white_pixels / total_pixels if total_pixels > 0 else 0

        # Calculate initial white percentage
        init_percentage = compute_white_percentage(corners_updated)

        while True:
            # Make a copy for testing
            test_corners = corners_updated.copy()

            # Adjust left edge
            test_corners[upper_left_idx, 0] += dx
            test_corners[lower_left_idx, 0] -= dx

            # Adjust right edge
            test_corners[upper_right_idx, 0] -= dx
            test_corners[lower_right_idx, 0] += dx

            # Calculate new white percentage
            new_percentage = compute_white_percentage(test_corners)

            # If percentage decreased, stop
            if new_percentage <= init_percentage * threshold_refine:
                break
                
            # Otherwise, keep the changes and continue
            corners_updated = test_corners

        return corners_updated

    
    def _detect_inner_from_brightness(self, frame: np.ndarray, 
                                      debug: bool = False) -> Optional[np.ndarray]:
        """Detect inner board using brightness mask (white/bright area)"""
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 3: Otsu threshold to get bright areas
        th_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        th_adjusted = th_otsu * 1.1
        _, mask = cv2.threshold(blurred, th_adjusted, 255, cv2.THRESH_BINARY)
        
        # Step 4: Morphological opening (remove noise)
        kernel = np.ones((7, 7), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Step 5: Find contours
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if debug:
                print("  ⚠️ No contours found in brightness mask")
            return None
        
        # Step 6: Get the largest bright contour (should be the board)
        largest = max(contours, key=cv2.contourArea)
        
        # Check if area is reasonable
        area = cv2.contourArea(largest)
        frame_area = frame.shape[0] * frame.shape[1]
        area_percent = area / frame_area * 100
        
        if debug:
            print(f"  Largest bright area: {area_percent:.1f}% of frame (need ≥12%)")
        
        if area < frame_area * 0.12:
            if debug:
                print(f"  ⚠️ Detected area too small: {area_percent:.1f}%")
            return None
        
        x, y, w, h = cv2.boundingRect(largest)
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)

        corners_updated = self.find_edges_with_white_threshold(mask_cleaned, corners, dx=5, threshold_white=0.8)
        corners_refined = self.refine_both_edges(mask_cleaned, corners_updated, dx=5, threshold_refine=1.001)
        
        # Order corners properly
        ordered_corners = self._order_points(corners_refined)
        
        return ordered_corners
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points as: top-left, top-right, bottom-right, bottom-left"""
        
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum: top-left has smallest sum, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        # Diff: top-right has smallest diff, bottom-left has largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def smooth_detection(self, corners: np.ndarray, window: int = 10) -> np.ndarray:
        """Smooth board detection over multiple frames"""
        
        self.board_history.append(corners)
        
        if len(self.board_history) > window:
            self.board_history.pop(0)
        
        return np.mean(self.board_history, axis=0)
    
    def board_movement_distance(self, corners1, corners2):
        movements = corners2 - corners1
        mean_movement = np.mean(movements, axis=0)
        deviations = movements - mean_movement
    
        # Weight Y deviations more heavily than X deviations
        y_weight = 3.0  # Adjust this value to control how much more Y matters
        weighted_deviations = deviations * np.array([1.0, y_weight])
    
        inconsistency = np.sum(weighted_deviations ** 2)
    
        return inconsistency
    
    
    def get_detection_data(self, frame: np.ndarray, threshold: int = 200) -> dict:
        """Get all intermediate detection data for visualization"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        th_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        th_adjusted = th_otsu * 1.1
        _, mask = cv2.threshold(blurred, th_adjusted, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((7, 7), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea) if contours else None
        
        area_percent = 0
        if largest is not None:
            area = cv2.contourArea(largest)
            frame_area = frame.shape[0] * frame.shape[1]
            area_percent = area / frame_area * 100
        
        x, y, w, h = cv2.boundingRect(largest)
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)

        corners_updated = self.find_edges_with_white_threshold(mask_cleaned, corners, dx=5, threshold_white=0.8)

        corners_final = self._detect_inner_from_brightness(frame, threshold)
        
        return {
            'gray': gray,
            'blurred': blurred,
            'mask': mask,
            'mask_cleaned': mask_cleaned,
            'contours': contours,
            'largest': largest,
            'corners_updated': corners_updated,
            'corners_final': corners_final,
            'threshold': threshold,
            'area_percent': area_percent
        }
    
    def process_video(self, output_path: str = None, visualize: bool = True,
                     show_cell_labels: bool = False, 
                     brightness_threshold: int = 200,
                     visualizer=None):
        """Process entire video and detect board in each frame"""
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            print(f"✗ Failed to open video: {self.video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing: {self.video_path.name}")
        print(f"Frames: {total_frames}, Resolution: {width}x{height}, FPS: {fps}")
        print(f"Detection: Brightness mask (threshold={brightness_threshold})")
        print(f"Grid: {self.grid_cols}x{self.grid_rows} adaptive rectangular cells")
        
        out = None
        if output_path and visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Calibrate with first frame
        ret, first_frame = cap.read()
        if not ret:
            print("✗ Failed to read first frame")
            return
        
        print("\nCalibrating...")
        print("Detecting inner board...")
        
        inner = self._detect_inner_from_brightness(
            first_frame, 
            debug=True
        )
        
        if inner is None:
            print("\n✗ Failed to detect inner board")
            print("\nTROUBLESHOOTING:")
            print("1. Check visualization outputs for analysis")
            print("2. If the board is darker, try lowering brightness_threshold (e.g., 150 or 120)")
            print("3. If the board is brighter, try increasing it (e.g., 220)")
            print("4. Make sure the playable board area is significantly brighter than surroundings")
            return
        
        self.board_history.append(inner)
        
        self.inner_board = inner
        print("\n✓ Board detection calibrated")
        print(f"  Inner board corners: {inner.astype(int).tolist()}")
        
        # Process all frames
        results = []
        frame_count = 0
        detected_count = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # threshold for maximum sum of distances between frame corners and previous frames corners to decide that board detected in a frame is valid
        init_threshold = 40
        # if we lose track of board, with each frame we multiplt htreshold by this constant to find board back
        threshold_multiplier = 1.1
        detection_threshold = init_threshold
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect inner board
            inner_corners = self._detect_inner_from_brightness(
                frame, 
                debug=False
            )
            
            if inner_corners is not None:
                detected_count += 1

                last_corners_mean = np.mean(self.board_history, axis=0)
                dist_corners = self.board_movement_distance(inner_corners, last_corners_mean)

                if dist_corners < detection_threshold:
                    inner_corners = self.smooth_detection(inner_corners)
                    detection_threshold = init_threshold
                else:
                    inner_corners = np.mean(self.board_history, axis=0)
                    detection_threshold *= threshold_multiplier
            else:
                # Use last known position
                inner_corners = np.mean(self.board_history, axis=0)
            
            # Store result
            results.append({
                'frame': frame_count,
                'detected': inner_corners is not None,
                'corners': inner_corners.tolist() if inner_corners is not None else None
            })
            
            # Visualize
            if visualize and visualizer and inner_corners is not None:
                vis_frame = visualizer.draw_detection_on_frame(
                    frame, inner_corners, frame_count, total_frames, 
                    detected_count, show_cell_labels, self.grid_cols, self.grid_rows
                )
                
                if out:
                    out.write(vis_frame)
            
            if frame_count % 100 == 0:
                print(f"Processed: {frame_count}/{total_frames} frames "
                      f"({detected_count}/{frame_count} detected)")
        
        cap.release()
        if out:
            out.release()
        
        print(f"\n{'='*60}")
        print(f"DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total frames: {frame_count}")
        print(f"Detected: {detected_count} ({detected_count/frame_count*100:.1f}%)")
        print(f"{'='*60}")
        
        # Save results
        results_path = self.video_path.parent / f"{self.video_path.stem}_board_detection.json"
        with open(results_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_frames': frame_count,
                    'detected_frames': detected_count,
                    'detection_rate': detected_count/frame_count,
                    'grid_size': f'{self.grid_cols}x{self.grid_rows}',
                    'detection_method': 'brightness_mask',
                    'brightness_threshold': brightness_threshold
                },
                'frames': results
            }, f, indent=2)
        
        print(f"✓ Results saved: {results_path}")
        return results