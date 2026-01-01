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
    
    def _detect_inner_from_brightness(self, frame: np.ndarray, 
                                      threshold: int = 200,
                                      debug: bool = False) -> Optional[np.ndarray]:
        """Detect inner board using brightness mask (white/bright area)"""
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 3: Threshold to get bright areas
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
        # Step 4: Morphological closing (fill holes)
        kernel = np.ones((7, 7), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Step 5: Morphological opening (remove noise)
        mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Step 6: Find contours
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if debug:
                print("  ⚠️ No contours found in brightness mask")
            return None
        
        # Step 7: Get the largest bright contour (should be the board)
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
        
        # Step 8: Approximate contour to quadrilateral
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
        
        # Order corners properly
        ordered_corners = self._order_points(corners)
        
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
    
    def smooth_detection(self, corners: np.ndarray, window: int = 5) -> np.ndarray:
        """Smooth board detection over multiple frames"""
        
        self.board_history.append(corners)
        
        if len(self.board_history) > window:
            self.board_history.pop(0)
        
        return np.mean(self.board_history, axis=0)
    
    def get_detection_data(self, frame: np.ndarray, threshold: int = 200) -> dict:
        """Get all intermediate detection data for visualization"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((7, 7), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea) if contours else None
        
        area_percent = 0
        if largest is not None:
            area = cv2.contourArea(largest)
            frame_area = frame.shape[0] * frame.shape[1]
            area_percent = area / frame_area * 100
        
        corners = self._detect_inner_from_brightness(frame, threshold)
        
        return {
            'gray': gray,
            'blurred': blurred,
            'mask': mask,
            'mask_closed': mask_closed,
            'mask_cleaned': mask_cleaned,
            'contours': contours,
            'largest': largest,
            'corners': corners,
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
            brightness_threshold, 
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
        
        self.inner_board = inner
        print("\n✓ Board detection calibrated")
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
            
            # Detect inner board
            inner_corners = self._detect_inner_from_brightness(
                frame, 
                brightness_threshold
            )
            
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