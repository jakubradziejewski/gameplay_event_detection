import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import json


class BoardDetector:
    """Detects and tracks Summoner Wars game board across video frames"""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.board_history = []  # Store detected boards for smoothing
        self.board_template = None
        
    def calibrate(self, frame: np.ndarray, manual_corners: Optional[np.ndarray] = None):
        """Calibrate detector with first frame or manual selection"""
        if manual_corners is not None:
            self.board_template = self._extract_board(frame, manual_corners)
            return manual_corners
        
        # Auto-detect in calibration frame
        return self._detect_multi_method(frame)
    
    def _detect_multi_method(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Try multiple detection methods and pick best result"""
        
        methods = [
            ("Color", self._detect_by_color),
            ("Edges", self._detect_by_edges),
            ("Contour", self._detect_by_contour)
        ]
        
        results = []
        for name, method in methods:
            try:
                corners = method(frame)
                if corners is not None and self._validate_board(corners, frame.shape):
                    results.append((name, corners, self._score_detection(corners, frame)))
            except Exception as e:
                print(f"Method {name} failed: {e}")
        
        if results:
            # Pick best scoring result
            results.sort(key=lambda x: x[2], reverse=True)
            best_name, best_corners, best_score = results[0]
            print(f"✓ Best detection: {best_name} (score: {best_score:.2f})")
            return best_corners
        
        return None
    
    def _detect_by_color(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect board using color segmentation"""
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Try multiple color ranges (adjust for your board)
        color_ranges = [
            # Beige/tan (common for game boards)
            ([10, 30, 100], [30, 150, 255]),
            # Blue tones
            ([90, 50, 50], [130, 255, 255]),
            # Green tones
            ([35, 40, 40], [85, 255, 255])
        ]
        
        masks = []
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            masks.append(mask)
        
        # Combine masks
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Clean up
        kernel = np.ones((7, 7), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            
            # Approximate to quadrilateral
            perimeter = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                return self._order_points(approx.reshape(4, 2))
            
            # Fallback: use minimum area rectangle
            rect = cv2.minAreaRect(largest)
            corners = cv2.boxPoints(rect)
            return self._order_points(corners)
        
        return None
    
    def _detect_by_edges(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect board using edge detection"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=frame.shape[1]//4, maxLineGap=20)
        
        if lines is None or len(lines) < 4:
            return None
        
        # Group lines
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            if angle < 15 or angle > 165:
                h_lines.append(line[0])
            elif 75 < angle < 105:
                v_lines.append(line[0])
        
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            return self._lines_to_corners(h_lines, v_lines)
        
        return None
    
    def _detect_by_contour(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect board by finding largest quadrilateral contour"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Looking for quadrilaterals
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                # Should be reasonably large
                if area > (frame.shape[0] * frame.shape[1] * 0.2):
                    return self._order_points(approx.reshape(4, 2))
        
        return None
    
    def _lines_to_corners(self, h_lines: List, v_lines: List) -> np.ndarray:
        """Convert horizontal and vertical lines to corner points"""
        
        # Find extreme lines
        h_lines = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        v_lines = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)
        
        top = h_lines[0]
        bottom = h_lines[-1]
        left = v_lines[0]
        right = v_lines[-1]
        
        # Calculate intersections
        top_y = (top[1] + top[3]) / 2
        bottom_y = (bottom[1] + bottom[3]) / 2
        left_x = (left[0] + left[2]) / 2
        right_x = (right[0] + right[2]) / 2
        
        corners = np.array([
            [left_x, top_y],
            [right_x, top_y],
            [right_x, bottom_y],
            [left_x, bottom_y]
        ], dtype=np.float32)
        
        return corners
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points as: top-left, top-right, bottom-right, bottom-left"""
        
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference to find corners
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def _validate_board(self, corners: np.ndarray, frame_shape: Tuple) -> bool:
        """Validate detected board makes sense"""
        
        if corners is None or len(corners) != 4:
            return False
        
        # Check if corners are within frame
        h, w = frame_shape[:2]
        if np.any(corners < 0) or np.any(corners[:, 0] > w) or np.any(corners[:, 1] > h):
            return False
        
        # Check minimum area (board should be at least 20% of frame)
        area = cv2.contourArea(corners.astype(np.int32))
        min_area = (h * w) * 0.2
        if area < min_area:
            return False
        
        # Check aspect ratio (should be roughly square)
        rect = cv2.minAreaRect(corners.astype(np.float32))
        w_rect, h_rect = rect[1]
        if w_rect == 0 or h_rect == 0:
            return False
        
        aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
        if aspect_ratio > 1.5:  # Allow some tolerance
            return False
        
        return True
    
    def _score_detection(self, corners: np.ndarray, frame: np.ndarray) -> float:
        """Score detection quality (0-1)"""
        
        score = 0.0
        
        # Size score (prefer larger detections)
        area = cv2.contourArea(corners.astype(np.int32))
        frame_area = frame.shape[0] * frame.shape[1]
        size_score = min(area / (frame_area * 0.8), 1.0)
        score += size_score * 0.4
        
        # Squareness score
        rect = cv2.minAreaRect(corners.astype(np.float32))
        w, h = rect[1]
        if w > 0 and h > 0:
            aspect = max(w, h) / min(w, h)
            squareness = 1.0 / aspect
            score += squareness * 0.3
        
        # Edge strength score
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [corners.astype(np.int32)], -1, 255, 2)
        edges = cv2.Canny(gray, 50, 150)
        edge_overlap = cv2.bitwise_and(mask, edges)
        edge_score = np.sum(edge_overlap > 0) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
        score += edge_score * 0.3
        
        return score
    
    def _extract_board(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Extract and warp board to top-down view"""
        
        # Define output size (square)
        size = 800
        dst_pts = np.array([
            [0, 0],
            [size-1, 0],
            [size-1, size-1],
            [0, size-1]
        ], dtype=np.float32)
        
        # Get perspective transform
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pts)
        warped = cv2.warpPerspective(frame, M, (size, size))
        
        return warped
    
    def smooth_detection(self, corners: np.ndarray, window: int = 5) -> np.ndarray:
        """Smooth board detection over multiple frames"""
        
        self.board_history.append(corners)
        
        # Keep only recent history
        if len(self.board_history) > window:
            self.board_history.pop(0)
        
        # Average corners
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
        
        # Prepare output video
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
        board_corners = self.calibrate(first_frame)
        
        if board_corners is None:
            print("✗ Failed to detect board in first frame")
            return
        
        print("✓ Board detected successfully")
        
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
            
            # Try to detect board
            corners = self._detect_multi_method(frame)
            
            if corners is not None:
                detected_count += 1
                corners = self.smooth_detection(corners)
            else:
                # Use last known position
                if self.board_history:
                    corners = self.board_history[-1]
            
            # Store result
            results.append({
                'frame': frame_count,
                'detected': corners is not None,
                'corners': corners.tolist() if corners is not None else None
            })
            
            # Visualize
            if visualize and corners is not None:
                vis_frame = frame.copy()
                cv2.polylines(vis_frame, [corners.astype(np.int32)], True, (0, 255, 0), 3)
                
                # Draw corner points
                for i, corner in enumerate(corners):
                    cv2.circle(vis_frame, tuple(corner.astype(int)), 8, (0, 0, 255), -1)
                    cv2.putText(vis_frame, str(i), tuple(corner.astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show progress
                progress = f"Frame: {frame_count}/{total_frames} ({detected_count}/{frame_count} detected)"
                cv2.putText(vis_frame, progress, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if out:
                    out.write(vis_frame)
            
            # Progress indicator
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
            json.dump({'summary': {
                'total_frames': frame_count,
                'detected_frames': detected_count,
                'detection_rate': detected_count/frame_count
            }, 'frames': results}, f, indent=2)
        
        print(f"✓ Results saved: {results_path}")
        
        return results


# Usage example
if __name__ == "__main__":
    detector = BoardDetector("data/easy/1.mp4")
    results = detector.process_video(
        output_path="output/board_detection_visualization.mp4",
        visualize=True
    )