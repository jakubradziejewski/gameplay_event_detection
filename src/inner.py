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
        self.inner_board = None  # Bright playable area detected from mask
        self.grid_cols = 8
        self.grid_rows = 6
        
    def calibrate(self, frame: np.ndarray, manual_corners: Optional[np.ndarray] = None):
        """Calibrate detector with first frame"""
        if manual_corners is not None:
            self.inner_board = manual_corners
            return manual_corners
        
        # Detect inner board from brightness mask
        inner = self._detect_inner_from_brightness(frame)
        self.inner_board = inner
        return inner
    
    def _detect_inner_from_brightness(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect inner board using brightness mask (white/bright area)"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to get bright areas (the playable board is much brighter)
        # Adjust threshold value based on your video - typically 150-200 works well
        _, mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask with morphological operations
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠ Could not detect bright area in frame")
            return None
        
        # Get the largest bright contour (should be the board)
        largest = max(contours, key=cv2.contourArea)
        
        # Check if area is reasonable
        area = cv2.contourArea(largest)
        frame_area = frame.shape[0] * frame.shape[1]
        if area < frame_area * 0.2:  # At least 20% of frame
            print(f"⚠ Detected area too small: {area / frame_area * 100:.1f}% of frame")
            return None
        
        # Approximate contour to quadrilateral
        perimeter = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            # Use bounding rectangle if not exactly 4 corners
            x, y, w, h = cv2.boundingRect(largest)
            corners = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)
        
        # Order corners
        return self._order_points(corners)
    
    def _draw_adaptive_rectangular_grid(self, frame: np.ndarray, corners: np.ndarray, 
                                        color: tuple = (0, 255, 255), 
                                        show_labels: bool = False,
                                        line_thickness: int = 1):
        """
        Draw 8x6 grid with rectangular cells that adapt to perspective
        Calculates dimensions dynamically based on corner positions
        """
        
        # Order: TL, TR, BR, BL
        tl, tr, br, bl = corners
        
        # Calculate vectors for interpolation
        # Top and bottom edges
        top_vec = tr - tl
        bottom_vec = br - bl
        
        # Left and right edges
        left_vec = bl - tl
        right_vec = br - tr
        
        # Draw vertical lines (9 lines for 8 columns)
        for col in range(self.grid_cols + 1):
            # Interpolation factor (0 to 1)
            t = col / self.grid_cols
            
            # Calculate top and bottom points for this vertical line
            top_pt = tl + top_vec * t
            bottom_pt = bl + bottom_vec * t
            
            cv2.line(frame, 
                    tuple(top_pt.astype(int)), 
                    tuple(bottom_pt.astype(int)), 
                    color, 
                    2 if col == 0 or col == self.grid_cols else line_thickness,
                    cv2.LINE_AA)
        
        # Draw horizontal lines (7 lines for 6 rows)
        for row in range(self.grid_rows + 1):
            # Interpolation factor (0 to 1)
            t = row / self.grid_rows
            
            # Calculate left and right points for this horizontal line
            left_pt = tl + left_vec * t
            right_pt = tr + right_vec * t
            
            cv2.line(frame, 
                    tuple(left_pt.astype(int)), 
                    tuple(right_pt.astype(int)), 
                    color,
                    2 if row == 0 or row == self.grid_rows else line_thickness,
                    cv2.LINE_AA)
        
        # Draw cell labels (A1-H6 style)
        if show_labels:
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    # Calculate cell center using bilinear interpolation
                    t_x = (col + 0.5) / self.grid_cols
                    t_y = (row + 0.5) / self.grid_rows
                    
                    # Interpolate along top and bottom edges
                    top_pt = tl + top_vec * t_x
                    bottom_pt = bl + bottom_vec * t_x
                    
                    # Interpolate between top and bottom
                    center = top_pt + (bottom_pt - top_pt) * t_y
                    
                    # Label as A1, B1, ..., H6
                    label = f"{chr(65 + col)}{row + 1}"
                    cv2.putText(frame, label, tuple(center.astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
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
    
    def smooth_detection(self, corners: np.ndarray, window: int = 5) -> np.ndarray:
        """Smooth board detection over multiple frames"""
        
        self.board_history.append(corners)
        
        if len(self.board_history) > window:
            self.board_history.pop(0)
        
        return np.mean(self.board_history, axis=0)
    
    def visualize_brightness_mask(self, frame: np.ndarray, output_path: str = "brightness_mask.jpg"):
        """Visualize the brightness mask for debugging"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((7, 7), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Create visualization
        vis = np.hstack([
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)
        ])
        
        cv2.putText(vis, "Original Grayscale", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, "Raw Threshold", (gray.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, "Cleaned Mask", (gray.shape[1] * 2 + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, vis)
        print(f"✓ Brightness mask visualization saved: {output_path}")
    
    def process_video(self, output_path: str = None, visualize: bool = True,
                     show_cell_labels: bool = False, 
                     save_brightness_debug: bool = False,
                     brightness_threshold: int = 150):
        """Process entire video and detect board in each frame using brightness mask"""
        
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
        print(f"Detection method: Brightness mask (threshold={brightness_threshold})")
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
        
        # Save brightness mask visualization if requested
        if save_brightness_debug:
            self.visualize_brightness_mask(first_frame, "output/brightness_mask_debug.jpg")
        
        print("  Detecting inner board from brightness mask...")
        inner = self._detect_inner_from_brightness(first_frame)
        
        if inner is None:
            print("✗ Failed to detect inner board")
            return
        
        self.inner_board = inner
        
        print("✓ Board detection calibrated")
        print(f"  Inner board corners: {inner.astype(int).tolist()}")
        
        # Process all frames
        results = []
        frame_count = 0
        detected_count = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect inner board from brightness
            inner_corners = self._detect_inner_from_brightness(frame)
            
            if inner_corners is not None:
                detected_count += 1
                inner_corners = self.smooth_detection(inner_corners)
            else:
                # Use last known position
                inner_corners = self.board_history[-1] if self.board_history else None
            
            # Store result
            results.append({
                'frame': frame_count,
                'detected': inner_corners is not None,
                'corners': inner_corners.tolist() if inner_corners is not None else None
            })
            
            # Visualize
            if visualize and inner_corners is not None:
                vis_frame = frame.copy()
                
                # Draw inner board boundary (green)
                cv2.polylines(vis_frame, [inner_corners.astype(np.int32)], 
                            True, (0, 255, 0), 3)
                
                # Draw adaptive rectangular grid
                self._draw_adaptive_rectangular_grid(vis_frame, inner_corners, 
                                                    (0, 255, 255), show_cell_labels)
                
                # Draw corners
                corner_labels = ['TL', 'TR', 'BR', 'BL']
                for corner, label in zip(inner_corners, corner_labels):
                    cv2.circle(vis_frame, tuple(corner.astype(int)), 8, (0, 0, 255), -1)
                    cv2.circle(vis_frame, tuple(corner.astype(int)), 10, (255, 255, 255), 2)
                    cv2.putText(vis_frame, label, tuple((corner + 15).astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show progress
                progress = f"Frame: {frame_count}/{total_frames} ({detected_count}/{frame_count} detected)"
                cv2.putText(vis_frame, progress, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(vis_frame, f"INNER BOARD ({self.grid_cols}x{self.grid_rows})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
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
                    'board_type': 'inner_playable_area',
                    'detection_method': 'brightness_mask',
                    'brightness_threshold': brightness_threshold
                },
                'frames': results
            }, f, indent=2)
        
        print(f"✓ Results saved: {results_path}")
        
        return results


# Usage
if __name__ == "__main__":
    detector = BoardDetector("data/easy/game_video.mp4")
    
    # Process video with brightness-based detection and adaptive grid
    results = detector.process_video(
        output_path="output/board_detection_brightness.mp4",
        visualize=True,
        show_cell_labels=False,  # Set True to show A1-H6 labels
        save_brightness_debug=True,  # Creates brightness_mask_debug.jpg
        brightness_threshold=150  # Adjust based on your video (120-200 range)
    )