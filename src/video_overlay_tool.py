import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import json


class VideoOverlayTool:
    """Add board detection visualization with 8x6 grid"""
    
    def __init__(self, video_path: str, json_path: str):
        self.video_path = Path(video_path)
        self.json_path = Path(json_path)
        self.detection_data = None
        self.grid_cols = 8  # Horizontal
        self.grid_rows = 6  # Vertical
        
    def load_json(self) -> bool:
        """Load detection results from JSON"""
        if not self.json_path.exists():
            print(f"✗ JSON file not found: {self.json_path}")
            return False
        
        with open(self.json_path, 'r') as f:
            self.detection_data = json.load(f)
        
        print(f"✓ Loaded detection data: {self.json_path.name}")
        print(f"  Total frames: {self.detection_data['summary']['total_frames']}")
        print(f"  Detection rate: {self.detection_data['summary']['detection_rate']*100:.1f}%")
        print(f"  Grid: {self.grid_cols}x{self.grid_rows} (rectangular cells)")
        
        return True
    
    def create_visualization(self, output_path: str, 
                           show_corners: bool = True,
                           show_grid: bool = True,
                           show_stats: bool = True,
                           show_cell_labels: bool = False,
                           color: tuple = (0, 255, 0),
                           thickness: int = 3):
        """Create video with 8x6 grid overlay"""
        
        if not self.load_json():
            return
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"✗ Failed to open video: {self.video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n{'='*60}")
        print(f"CREATING VISUALIZATION (8x6 GRID)")
        print(f"{'='*60}")
        print(f"Input: {self.video_path.name}")
        print(f"Output: {output_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        print(f"{'='*60}\n")
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < len(self.detection_data['frames']):
                frame_data = self.detection_data['frames'][frame_idx]
                
                if frame_data['detected'] and frame_data['corners']:
                    corners = np.array(frame_data['corners'], dtype=np.float32)
                    
                    # Draw board outline
                    cv2.polylines(frame, [corners.astype(np.int32)], 
                                True, color, thickness)
                    
                    # Draw corners
                    if show_corners:
                        corner_labels = ['TL', 'TR', 'BR', 'BL']
                        for i, (corner, label) in enumerate(zip(corners, corner_labels)):
                            x, y = corner.astype(int)
                            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
                            cv2.putText(frame, label, (x + 15, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                      (255, 255, 255), 2)
                    
                    # Draw 8x6 grid
                    if show_grid:
                        self._draw_8x6_grid(frame, corners, 
                                          show_labels=show_cell_labels,
                                          color=(0, 255, 255))
                    
                    status_color = (0, 255, 0)
                    status_text = "INNER BOARD DETECTED"
                else:
                    status_color = (0, 0, 255)
                    status_text = "NO DETECTION"
                
                if show_stats:
                    self._draw_stats(frame, frame_idx + 1, total_frames, 
                                   status_text, status_color,
                                   self.detection_data['summary'])
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {frame_idx}/{total_frames} ({progress:.1f}%)")
        
        cap.release()
        out.release()
        
        print(f"\n{'='*60}")
        print(f"✓ Visualization complete!")
        print(f"  Processed: {frame_idx} frames")
        print(f"  Output: {output_path}")
        print(f"{'='*60}\n")
    
    def _draw_8x6_grid(self, frame: np.ndarray, corners: np.ndarray,
                       show_labels: bool = False, color: tuple = (0, 255, 255)):
        """Draw 8x6 grid (8 columns, 6 rows) on detected board"""
        
        # Define grid dimensions
        grid_width = 800
        grid_height = 600  # 6 rows instead of 8
        
        # Destination corners for warping
        dst_corners = np.array([
            [0, 0],
            [grid_width, 0],
            [grid_width, grid_height],
            [0, grid_height]
        ], dtype=np.float32)
        
        # Get perspective transform
        M = cv2.getPerspectiveTransform(dst_corners, corners)
        
        # Calculate cell dimensions
        cell_width = grid_width / self.grid_cols
        cell_height = grid_height / self.grid_rows
        
        # Draw vertical lines (8 columns = 9 lines)
        for i in range(self.grid_cols + 1):
            x = i * cell_width
            pt1 = np.array([[x, 0]], dtype=np.float32)
            pt2 = np.array([[x, grid_height]], dtype=np.float32)
            
            pt1_t = cv2.perspectiveTransform(pt1.reshape(1, 1, 2), M)[0][0]
            pt2_t = cv2.perspectiveTransform(pt2.reshape(1, 1, 2), M)[0][0]
            
            # Thicker lines for borders
            line_thickness = 2 if i == 0 or i == self.grid_cols else 1
            cv2.line(frame, tuple(pt1_t.astype(int)), 
                    tuple(pt2_t.astype(int)), color, line_thickness, cv2.LINE_AA)
        
        # Draw horizontal lines (6 rows = 7 lines)
        for i in range(self.grid_rows + 1):
            y = i * cell_height
            pt1 = np.array([[0, y]], dtype=np.float32)
            pt2 = np.array([[grid_width, y]], dtype=np.float32)
            
            pt1_t = cv2.perspectiveTransform(pt1.reshape(1, 1, 2), M)[0][0]
            pt2_t = cv2.perspectiveTransform(pt2.reshape(1, 1, 2), M)[0][0]
            
            line_thickness = 2 if i == 0 or i == self.grid_rows else 1
            cv2.line(frame, tuple(pt1_t.astype(int)), 
                    tuple(pt2_t.astype(int)), color, line_thickness, cv2.LINE_AA)
        
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
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def _draw_stats(self, frame: np.ndarray, current_frame: int, 
                    total_frames: int, status_text: str, 
                    status_color: tuple, summary: Dict):
        """Draw statistics overlay"""
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        y_pos = 35
        line_height = 30
        
        # Frame info
        cv2.putText(frame, f"Frame: {current_frame}/{total_frames}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height
        
        # Progress
        progress = (current_frame / total_frames) * 100
        cv2.putText(frame, f"Progress: {progress:.1f}%", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height
        
        # Status
        cv2.putText(frame, f"Status: {status_text}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        y_pos += line_height
        
        # Detection rate
        rate = summary['detection_rate'] * 100
        cv2.putText(frame, f"Detection Rate: {rate:.1f}%", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height
        
        # Grid info
        cv2.putText(frame, f"Grid: 8x6 (Rectangular cells)", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def create_side_by_side(self, output_path: str, show_labels: bool = False):
        """Create side-by-side comparison"""
        
        if not self.load_json():
            return
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        print(f"\nCreating side-by-side comparison (8x6 grid)...")
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            original = frame.copy()
            annotated = frame.copy()
            
            if frame_idx < len(self.detection_data['frames']):
                frame_data = self.detection_data['frames'][frame_idx]
                
                if frame_data['detected'] and frame_data['corners']:
                    corners = np.array(frame_data['corners'], dtype=np.float32)
                    
                    cv2.polylines(annotated, [corners.astype(np.int32)], 
                                True, (0, 255, 0), 3)
                    
                    for corner in corners:
                        cv2.circle(annotated, tuple(corner.astype(int)), 
                                 8, (0, 0, 255), -1)
                    
                    self._draw_8x6_grid(annotated, corners, show_labels=show_labels)
            
            cv2.putText(original, "ORIGINAL", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated, "INNER BOARD (8x6)", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            combined = np.hstack([original, annotated])
            out.write(combined)
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed: {frame_idx} frames")
        
        cap.release()
        out.release()
        
        print(f"✓ Side-by-side video saved: {output_path}\n")
    
    def extract_warped_boards(self, output_dir: str, sample_rate: int = 30):
        """Extract warped 8x6 board views"""
        
        if not self.load_json():
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return
        
        print(f"\nExtracting warped board views (8x6 aspect ratio)...")
        print(f"Sample rate: every {sample_rate} frames")
        
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                if frame_idx < len(self.detection_data['frames']):
                    frame_data = self.detection_data['frames'][frame_idx]
                    
                    if frame_data['detected'] and frame_data['corners']:
                        corners = np.array(frame_data['corners'], dtype=np.float32)
                        
                        # Warp with 8:6 aspect ratio
                        warped = self._warp_8x6(frame, corners)
                        
                        output_file = output_path / f"board_frame_{frame_idx:05d}.jpg"
                        cv2.imwrite(str(output_file), warped)
                        saved_count += 1
            
            frame_idx += 1
        
        cap.release()
        
        print(f"✓ Extracted {saved_count} warped board images (800x600)\n")
    
    def _warp_8x6(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Warp board to top-down view with 8:6 aspect ratio"""
        
        width = 800
        height = 600  # Maintains 8:6 aspect ratio
        
        dst_corners = np.array([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(corners, dst_corners)
        warped = cv2.warpPerspective(frame, M, (width, height))
        
        return warped


if __name__ == "__main__":
    print("="*60)
    print("VIDEO OVERLAY TOOL - 8x6 GRID")
    print("="*60)
    
    tool = VideoOverlayTool(
        video_path="data/easy/game_video.mp4",
        json_path="data/easy/game_video_board_detection.json"
    )
    
    # Create visualization with 8x6 grid
    tool.create_visualization(
        output_path="game_with_8x6_grid.mp4",
        show_corners=True,
        show_grid=True,
        show_stats=True,
        show_cell_labels=False,  # Set True to show A1-H6 labels
        color=(0, 255, 0),
        thickness=3
    )
    
    # Optional: side-by-side with cell labels
    # tool.create_side_by_side("output/comparison_8x6.mp4", show_labels=True)
    
    # Optional: extract warped boards
    # tool.extract_warped_boards("output/boards_8x6", sample_rate=30)