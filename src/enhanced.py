import cv2
import numpy as np
from pathlib import Path
import json


class AdjustedGridOverlay:
    """Grid overlay with correct rectangular cells (height = 2x width)"""
    
    def __init__(self, video_path: str, json_path: str):
        self.video_path = Path(video_path)
        self.json_path = Path(json_path)
        self.detection_data = None
        self.grid_cols = 8
        self.grid_rows = 6
        
        # Correct padding ratios
        self.padding_left_right = 0.10  # 10% padding on sides
        self.padding_top_bottom = 0.025  # 2.5% padding on top/bottom (4x less)
    
    def load_json(self) -> bool:
        """Load detection results"""
        if not self.json_path.exists():
            print(f"✗ JSON not found: {self.json_path}")
            return False
        
        with open(self.json_path, 'r') as f:
            self.detection_data = json.load(f)
        
        print(f"✓ Loaded: {self.json_path.name}")
        return True
    
    def create_adjusted_visualization(self, output_path: str,
                                     show_both_boundaries: bool = True,
                                     show_cell_labels: bool = False):
        """Create visualization with correct rectangular cells"""
        
        if not self.load_json():
            return
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"✗ Failed to open: {self.video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\nCreating adjusted grid visualization...")
        print(f"Padding: Left/Right={self.padding_left_right*100}%, "
              f"Top/Bottom={self.padding_top_bottom*100}%")
        print(f"Grid: 8x6 with rectangular cells (height = 2x width)")
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < len(self.detection_data['frames']):
                frame_data = self.detection_data['frames'][frame_idx]
                
                if frame_data['detected'] and frame_data['corners']:
                    outer_corners = np.array(frame_data['corners'], dtype=np.float32)
                    
                    # Calculate inner corners with padding
                    inner_corners = self._apply_padding(outer_corners)
                    
                    if show_both_boundaries:
                        # Draw outer boundary (detected board) in red
                        cv2.polylines(frame, [outer_corners.astype(np.int32)], 
                                    True, (0, 0, 255), 2)
                        cv2.putText(frame, "OUTER (Detected)", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Draw inner boundary (playable area) in green
                        cv2.polylines(frame, [inner_corners.astype(np.int32)], 
                                    True, (0, 255, 0), 3)
                        cv2.putText(frame, "INNER (Playable 8x6)", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # Only draw inner
                        cv2.polylines(frame, [inner_corners.astype(np.int32)], 
                                    True, (0, 255, 0), 3)
                    
                    # Draw 8x6 grid with rectangular cells on INNER area
                    self._draw_rectangular_grid(frame, inner_corners, 
                                               (0, 255, 255), show_cell_labels)
                    
                    # Draw corner markers on inner area
                    labels = ['TL', 'TR', 'BR', 'BL']
                    for corner, label in zip(inner_corners, labels):
                        cv2.circle(frame, tuple(corner.astype(int)), 6, (255, 0, 255), -1)
                        cv2.putText(frame, label, tuple((corner + 12).astype(int)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Stats
            cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Progress: {frame_idx}/{total_frames}")
        
        cap.release()
        out.release()
        
        print(f"✓ Complete: {output_path}\n")
    
    def _apply_padding(self, outer_corners: np.ndarray) -> np.ndarray:
        """Apply padding to outer corners to get inner playable area"""
        
        # Order: TL, TR, BR, BL
        tl, tr, br, bl = outer_corners
        
        # Calculate vectors for each side
        top_vec = tr - tl
        bottom_vec = br - bl
        left_vec = bl - tl
        right_vec = br - tr
        
        # Apply padding (less on top/bottom, more on left/right)
        # Top-left: move right and down
        new_tl = tl + top_vec * self.padding_left_right + left_vec * self.padding_top_bottom
        
        # Top-right: move left and down
        new_tr = tr - top_vec * self.padding_left_right + right_vec * self.padding_top_bottom
        
        # Bottom-right: move left and up
        new_br = br - bottom_vec * self.padding_left_right - right_vec * self.padding_top_bottom
        
        # Bottom-left: move right and up
        new_bl = bl + bottom_vec * self.padding_left_right - left_vec * self.padding_top_bottom
        
        return np.array([new_tl, new_tr, new_br, new_bl], dtype=np.float32)
    
    def _draw_rectangular_grid(self, frame: np.ndarray, corners: np.ndarray, 
                               color: tuple, show_labels: bool = False):
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
        
        # Verify proportions
        ratio = cell_height / cell_width
        print(f"Cell ratio (height/width): {ratio:.2f} (target: 2.0)")
        
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
        
        for i, (pad_lr, pad_tb, label) in enumerate(padding_options):
            self.padding_left_right = pad_lr
            self.padding_top_bottom = pad_tb
            
            inner = self._apply_padding(outer_corners)
            cv2.polylines(vis, [inner.astype(np.int32)], True, colors[i], 2)
            
            # Label
            y_pos = 100 + i * 30
            cv2.putText(vis, label, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        
        # Draw outer
        cv2.polylines(vis, [outer_corners.astype(np.int32)], True, (0, 0, 255), 3)
        cv2.putText(vis, "OUTER (Detected)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(vis, "Ratio: L/R padding = 4x T/B padding", (10, vis.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, vis)
        print(f"✓ Padding calibration saved: {output_path}")
        print("  Review the image and adjust padding values if needed")


if __name__ == "__main__":
    tool = AdjustedGridOverlay(
        video_path="data/easy/game_video.mp4",
        json_path="data/easy/game_video_board_detection.json"
    )
    
    # Create visualization with correct rectangular cells
    tool.create_adjusted_visualization(
        output_path="output/adjusted_grid_rectangular.mp4",
        show_both_boundaries=True,  # Shows both outer and inner boundaries
        show_cell_labels=False  # Set True to show A1-H6 labels
    )
    
    # Optional: Create padding calibration image to fine-tune
    cap = cv2.VideoCapture("data/easy/game_video.mp4")
    ret, frame = cap.read()
    if ret and tool.load_json():
        first_detection = tool.detection_data['frames'][0]
        if first_detection['detected']:
            corners = np.array(first_detection['corners'], dtype=np.float32)
            tool.calibrate_padding(frame, corners, "output/padding_calibration.jpg")
    cap.release()