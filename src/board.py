import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import json


class BoardDetector:
    """Detects Summoner Wars game board (inner playable area with rectangular cells)"""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.board_history = []
        self.outer_board = None  # Dark border
        self.inner_board = None  # Bright playable area
        self.grid_cols = 8
        self.grid_rows = 6
        
        # Padding ratios for rectangular cells (height = 2x width)
        self.padding_left_right = 0.10  # 10% padding on sides
        self.padding_top_bottom = 0.025  # 2.5% padding on top/bottom (4x less)
        
    def calibrate(self, frame: np.ndarray, manual_corners: Optional[np.ndarray] = None):
        """Calibrate detector with first frame"""
        if manual_corners is not None:
            self.inner_board = manual_corners
            return manual_corners
        
        # First detect outer board, then find inner
        outer = self._detect_outer_board(frame)
        if outer is not None:
            self.outer_board = outer
            inner = self._apply_padding(outer)
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
    
    def _apply_padding(self, outer_corners: np.ndarray) -> np.ndarray:
        """Apply padding to outer corners to get inner playable area with rectangular cells"""
        
        # Order: TL, TR, BR, BL
        tl, tr, br, bl = outer_corners
        
        # Calculate vectors for each side
        top_vec = tr - tl
        bottom_vec = br - bl
        left_vec = bl - tl
        right_vec = br - tr
        
        # Apply padding (less on top/bottom, more on left/right for rectangular cells)
        # Top-left: move right and down
        new_tl = tl + top_vec * self.padding_left_right + left_vec * self.padding_top_bottom
        
        # Top-right: move left and down
        new_tr = tr - top_vec * self.padding_left_right + right_vec * self.padding_top_bottom
        
        # Bottom-right: move left and up
        new_br = br - bottom_vec * self.padding_left_right - right_vec * self.padding_top_bottom
        
        # Bottom-left: move right and up
        new_bl = bl + bottom_vec * self.padding_left_right - left_vec * self.padding_top_bottom
        
        return np.array([new_tl, new_tr, new_br, new_bl], dtype=np.float32)
    
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
    
    def _draw_rectangular_grid(self, frame: np.ndarray, corners: np.ndarray, 
                               color: tuple = (0, 255, 255), 
                               show_labels: bool = False):
        """
        Draw 8x6 grid with rectangular cells where height = 2x width
        
        For an 8x6 grid with rectangular cells (h=2w):
        - 8 columns across
        - 6 rows down
        - Each cell: width = 1 unit, height = 2 units
        - Total dimensions: width = 8 units, height = 12 units (6 rows × 2)
        """
        
        # Grid dimensions: width=8, height=12 (to make cells rectangular with h=2w)
        grid_width = 800
        grid_height = 1200  # 6 rows × 2 = 12 units of height
        
        dst = np.array([[0, 0], [grid_width, 0], [grid_width, grid_height], [0, grid_height]], 
                      dtype=np.float32)
        M = cv2.getPerspectiveTransform(dst, corners)
        
        # Cell dimensions
        cell_width = grid_width / self.grid_cols  # 100 pixels
        cell_height = grid_height / self.grid_rows  # 200 pixels (2x width)
        
        # Draw vertical lines (8 columns = 9 lines)
        for i in range(self.grid_cols + 1):
            x = i * cell_width
            pt1 = np.array([[x, 0]], dtype=np.float32)
            pt2 = np.array([[x, grid_height]], dtype=np.float32)
            
            pt1_t = cv2.perspectiveTransform(pt1.reshape(1, 1, 2), M)[0][0]
            pt2_t = cv2.perspectiveTransform(pt2.reshape(1, 1, 2), M)[0][0]
            
            thickness = 2 if i == 0 or i == self.grid_cols else 1
            cv2.line(frame, tuple(pt1_t.astype(int)), tuple(pt2_t.astype(int)), 
                    color, thickness, cv2.LINE_AA)
        
        # Draw horizontal lines (6 rows = 7 lines)
        for i in range(self.grid_rows + 1):
            y = i * cell_height
            pt1 = np.array([[0, y]], dtype=np.float32)
            pt2 = np.array([[grid_width, y]], dtype=np.float32)
            
            pt1_t = cv2.perspectiveTransform(pt1.reshape(1, 1, 2), M)[0][0]
            pt2_t = cv2.perspectiveTransform(pt2.reshape(1, 1, 2), M)[0][0]
            
            thickness = 2 if i == 0 or i == self.grid_rows else 1
            cv2.line(frame, tuple(pt1_t.astype(int)), tuple(pt2_t.astype(int)), 
                    color, thickness, cv2.LINE_AA)
        
        # Draw cell labels (A1-H6 style)
        if show_labels:
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    # Calculate cell center
                    center_x = (col + 0.5) * cell_width
                    center_y = (row + 0.5) * cell_height
                    
                    pt = np.array([[center_x, center_y]], dtype=np.float32)
                    pt_t = cv2.perspectiveTransform(pt.reshape(1, 1, 2), M)[0][0]
                    
                    # Label as A1, B1, etc.
                    label = f"{chr(65 + col)}{row + 1}"
                    cv2.putText(frame, label, tuple(pt_t.astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
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
    
    def calibrate_padding(self, frame: np.ndarray, outer_corners: np.ndarray,
                         output_path: str = "padding_calibration.jpg"):
        """
        Interactive tool to help find the right padding values.
        Draws multiple padding options for visual comparison.
        """
        
        padding_options = [
            (0.08, 0.020, "8% L/R, 2% T/B"),
            (0.10, 0.025, "10% L/R, 2.5% T/B (Default)"),
            (0.12, 0.030, "12% L/R, 3% T/B"),
            (0.15, 0.037, "15% L/R, 3.7% T/B"),
        ]
        
        vis = frame.copy()
        
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (255, 0, 255),  # Magenta
            (255, 128, 0),  # Orange
        ]
        
        # Save original padding values
        orig_lr = self.padding_left_right
        orig_tb = self.padding_top_bottom
        
        for i, (pad_lr, pad_tb, label) in enumerate(padding_options):
            self.padding_left_right = pad_lr
            self.padding_top_bottom = pad_tb
            
            inner = self._apply_padding(outer_corners)
            cv2.polylines(vis, [inner.astype(np.int32)], True, colors[i], 2)
            
            # Label
            y_pos = 100 + i * 30
            cv2.putText(vis, label, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        
        # Restore original values
        self.padding_left_right = orig_lr
        self.padding_top_bottom = orig_tb
        
        # Draw outer
        cv2.polylines(vis, [outer_corners.astype(np.int32)], True, (0, 0, 255), 3)
        cv2.putText(vis, "OUTER (Detected)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(vis, "Ratio: L/R padding = 4x T/B padding", (10, vis.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, vis)
        print(f"✓ Padding calibration saved: {output_path}")
        print("  Review the image and adjust padding values if needed")
    
    def process_video(self, output_path: str = None, visualize: bool = True,
                     show_both_boundaries: bool = True, show_cell_labels: bool = False):
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
        print(f"Padding: Left/Right={self.padding_left_right*100}%, "
              f"Top/Bottom={self.padding_top_bottom*100}%")
        print(f"Grid: {self.grid_cols}x{self.grid_rows} with rectangular cells (height = 2x width)")
        
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
        
        print("  Step 2: Applying padding for inner playable area...")
        inner = self._apply_padding(outer)
        
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
                # Apply padding to get inner
                inner_corners = self._apply_padding(outer_corners)
                detected_count += 1
                inner_corners = self.smooth_detection(inner_corners)
            else:
                # Use last known position
                inner_corners = self.board_history[-1] if self.board_history else None
                outer_corners = self.outer_board
            
            # Store result
            results.append({
                'frame': frame_count,
                'detected': inner_corners is not None,
                'corners': inner_corners.tolist() if inner_corners is not None else None,
                'outer_corners': outer_corners.tolist() if outer_corners is not None else None
            })
            
            # Visualize
            if visualize and inner_corners is not None:
                vis_frame = frame.copy()
                
                if show_both_boundaries and outer_corners is not None:
                    # Draw outer boundary (red)
                    cv2.polylines(vis_frame, [outer_corners.astype(np.int32)], 
                                True, (0, 0, 255), 2)
                    cv2.putText(vis_frame, "OUTER (Detected)", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw inner board (main - green)
                cv2.polylines(vis_frame, [inner_corners.astype(np.int32)], 
                            True, (0, 255, 0), 3)
                
                # Draw rectangular grid
                self._draw_rectangular_grid(vis_frame, inner_corners, 
                                           (0, 255, 255), show_cell_labels)
                
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
                    'padding_left_right': self.padding_left_right,
                    'padding_top_bottom': self.padding_top_bottom,
                    'cell_aspect_ratio': '2:1 (height:width)'
                },
                'frames': results
            }, f, indent=2)
        
        print(f"✓ Results saved: {results_path}")
        
        return results


# Usage
if __name__ == "__main__":
    detector = BoardDetector("data/easy/game_video.mp4")
    
    # Optional: Create padding calibration image first
    cap = cv2.VideoCapture("data/easy/game_video.mp4")
    ret, frame = cap.read()
    if ret:
        outer = detector._detect_outer_board(frame)
        if outer is not None:
            detector.calibrate_padding(frame, outer, "output/padding_calibration.jpg")
    cap.release()
    
    # Process video with rectangular grid
    results = detector.process_video(
        output_path="output/board_detection_rectangular.mp4",
        visualize=True,
        show_both_boundaries=True,  # Show both outer and inner boundaries
        show_cell_labels=False  # Set True to show A1-H6 labels
    )