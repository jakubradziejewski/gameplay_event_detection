import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import json


class BoardDetector:
    """Detects Summoner Wars game board (inner playable area)"""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.board_history = []
        self.outer_board = None  # Dark border
        self.inner_board = None  # Bright playable area
        
    def calibrate(self, frame: np.ndarray, manual_corners: Optional[np.ndarray] = None):
        """Calibrate detector with first frame"""
        if manual_corners is not None:
            self.inner_board = manual_corners
            return manual_corners
        
        # First detect outer board, then find inner
        outer = self._detect_outer_board(frame)
        if outer is not None:
            self.outer_board = outer
            inner = self._detect_inner_board(frame, outer)
            self.inner_board = inner
            return inner
        
        return None
    
    def _detect_outer_board(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect the outer board (including dark border)"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold to find board outline
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > (frame.shape[0] * frame.shape[1] * 0.3):
                    return self._order_points(approx.reshape(4, 2))
        
        # Fallback: edge detection method
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                return self._order_points(approx.reshape(4, 2))
            
            # Use bounding rectangle
            rect = cv2.minAreaRect(largest)
            corners = cv2.boxPoints(rect)
            return self._order_points(corners)
        
        return None
    
    def _detect_inner_board(self, frame: np.ndarray, 
                           outer_corners: np.ndarray) -> Optional[np.ndarray]:
        """Detect inner bright playable area from outer board"""
        
        # Warp outer board to top-down view for easier processing
        warped = self._warp_board(frame, outer_corners, size=1000)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        
        # Detect bright areas (the playable board is lighter)
        # Adjust these values based on your board's actual color
        lower_bright = np.array([0, 0, 120])  # Low saturation, high value
        upper_bright = np.array([180, 100, 255])
        
        mask_bright = cv2.inRange(hsv, lower_bright, upper_bright)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask_bright = cv2.morphologyEx(mask_bright, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_bright = cv2.morphologyEx(mask_bright, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find the largest bright region (inner board)
        contours, _ = cv2.findContours(mask_bright, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠ Could not detect inner board, using outer with margin")
            return self._apply_margin(outer_corners, margin_ratio=0.08)
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            inner_corners_warped = approx.reshape(4, 2).astype(np.float32)
        else:
            # Use bounding rectangle
            x, y, w, h = cv2.boundingRect(largest)
            inner_corners_warped = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)
        
        # Transform inner corners back to original frame coordinates
        inner_corners = self._unwarp_points(inner_corners_warped, outer_corners, 1000)
        
        return self._order_points(inner_corners)
    
    def _apply_margin(self, corners: np.ndarray, margin_ratio: float = 0.08) -> np.ndarray:
        """Apply margin to outer corners to estimate inner board"""
        
        # Calculate center
        center = np.mean(corners, axis=0)
        
        # Move each corner toward center by margin_ratio
        inner_corners = []
        for corner in corners:
            direction = center - corner
            new_corner = corner + direction * margin_ratio
            inner_corners.append(new_corner)
        
        return np.array(inner_corners, dtype=np.float32)
    
    def _warp_board(self, frame: np.ndarray, corners: np.ndarray, 
                    size: int = 1000) -> np.ndarray:
        """Warp board to top-down view"""
        
        dst_corners = np.array([
            [0, 0],
            [size-1, 0],
            [size-1, size-1],
            [0, size-1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(corners, dst_corners)
        warped = cv2.warpPerspective(frame, M, (size, size))
        
        return warped
    
    def _unwarp_points(self, points: np.ndarray, outer_corners: np.ndarray, 
                       size: int) -> np.ndarray:
        """Transform points from warped space back to original frame"""
        
        dst_corners = np.array([
            [0, 0],
            [size-1, 0],
            [size-1, size-1],
            [0, size-1]
        ], dtype=np.float32)
        
        # Inverse transform
        M = cv2.getPerspectiveTransform(dst_corners, outer_corners)
        
        # Transform points
        points_reshaped = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points_reshaped, M)
        
        return transformed.reshape(-1, 2)
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points as: top-left, top-right, bottom-right, bottom-left"""
        
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def _validate_board(self, corners: np.ndarray, frame_shape: Tuple) -> bool:
        """Validate detected board"""
        
        if corners is None or len(corners) != 4:
            return False
        
        h, w = frame_shape[:2]
        if np.any(corners < 0) or np.any(corners[:, 0] > w) or np.any(corners[:, 1] > h):
            return False
        
        area = cv2.contourArea(corners.astype(np.int32))
        min_area = (h * w) * 0.15
        if area < min_area:
            return False
        
        return True
    
    def smooth_detection(self, corners: np.ndarray, window: int = 5) -> np.ndarray:
        """Smooth board detection over multiple frames"""
        
        self.board_history.append(corners)
        
        if len(self.board_history) > window:
            self.board_history.pop(0)
        
        return np.mean(self.board_history, axis=0)
    
    def process_video(self, output_path: str = None, visualize: bool = True):
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
        print("  Step 1: Detecting outer board...")
        outer = self._detect_outer_board(first_frame)
        
        if outer is None:
            print("✗ Failed to detect outer board")
            return
        
        print("  Step 2: Detecting inner playable area...")
        inner = self._detect_inner_board(first_frame, outer)
        
        if inner is None:
            print("✗ Failed to detect inner board")
            return
        
        self.outer_board = outer
        self.inner_board = inner
        
        print("✓ Board detection calibrated")
        print(f"  Outer board: {outer.astype(int).tolist()}")
        print(f"  Inner board: {inner.astype(int).tolist()}")
        
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
            
            # Try to detect outer first
            outer_corners = self._detect_outer_board(frame)
            
            if outer_corners is not None:
                # Detect inner from outer
                inner_corners = self._detect_inner_board(frame, outer_corners)
                
                if inner_corners is not None:
                    detected_count += 1
                    inner_corners = self.smooth_detection(inner_corners)
                else:
                    inner_corners = self.board_history[-1] if self.board_history else None
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
                
                # Draw inner board (main - green)
                cv2.polylines(vis_frame, [inner_corners.astype(np.int32)], 
                            True, (0, 255, 0), 3)
                
                # Draw corners
                corner_labels = ['TL', 'TR', 'BR', 'BL']
                for i, (corner, label) in enumerate(zip(inner_corners, corner_labels)):
                    cv2.circle(vis_frame, tuple(corner.astype(int)), 8, (0, 0, 255), -1)
                    cv2.circle(vis_frame, tuple(corner.astype(int)), 10, (255, 255, 255), 2)
                    cv2.putText(vis_frame, label, tuple((corner + 15).astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show progress
                progress = f"Frame: {frame_count}/{total_frames} ({detected_count}/{frame_count} detected)"
                cv2.putText(vis_frame, progress, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(vis_frame, "INNER BOARD (8x6)", (10, 60),
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
                    'grid_size': '8x6',
                    'board_type': 'inner_playable_area'
                },
                'frames': results
            }, f, indent=2)
        
        print(f"✓ Results saved: {results_path}")
        
        return results


# Usage
if __name__ == "__main__":
    detector = BoardDetector("data/easy/game_video.mp4")
    results = detector.process_video(
        output_path="output/board_detection_inner.mp4",
        visualize=True
    )