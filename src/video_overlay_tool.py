import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import json


class VideoOverlayTool:
    """Add board detection visualization to video from JSON results"""
    
    def __init__(self, video_path: str, json_path: str):
        self.video_path = Path(video_path)
        self.json_path = Path(json_path)
        self.detection_data = None
        
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
        
        return True
    
    def create_visualization(self, output_path: str, 
                           show_corners: bool = True,
                           show_grid: bool = True,
                           show_stats: bool = True,
                           color: tuple = (0, 255, 0),
                           thickness: int = 3):
        """Create video with detection overlay"""
        
        if not self.load_json():
            return
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"✗ Failed to open video: {self.video_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n{'='*60}")
        print(f"CREATING VISUALIZATION")
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
            
            # Get detection data for this frame
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
                    
                    # Draw grid overlay (if board detected)
                    if show_grid:
                        self._draw_grid(frame, corners, color=(0, 255, 255))
                    
                    # Status indicator
                    status_color = (0, 255, 0)
                    status_text = "BOARD DETECTED"
                else:
                    status_color = (0, 0, 255)
                    status_text = "NO DETECTION"
                
                # Show stats overlay
                if show_stats:
                    self._draw_stats(frame, frame_idx + 1, total_frames, 
                                   status_text, status_color,
                                   self.detection_data['summary'])
            
            out.write(frame)
            frame_idx += 1
            
            # Progress
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
    
    def _draw_grid(self, frame: np.ndarray, corners: np.ndarray, 
                   grid_size: int = 8, color: tuple = (0, 255, 255)):
        """Draw grid overlay on detected board"""
        
        # Define destination points for grid
        dst_corners = np.array([
            [0, 0],
            [800, 0],
            [800, 800],
            [0, 800]
        ], dtype=np.float32)
        
        # Get perspective transform
        M = cv2.getPerspectiveTransform(dst_corners, corners)
        
        # Draw vertical and horizontal lines
        step = 800 // grid_size
        
        for i in range(1, grid_size):
            # Vertical lines
            pt1 = np.array([[i * step, 0]], dtype=np.float32)
            pt2 = np.array([[i * step, 800]], dtype=np.float32)
            
            pt1_t = cv2.perspectiveTransform(pt1.reshape(1, 1, 2), M)[0][0]
            pt2_t = cv2.perspectiveTransform(pt2.reshape(1, 1, 2), M)[0][0]
            
            cv2.line(frame, tuple(pt1_t.astype(int)), 
                    tuple(pt2_t.astype(int)), color, 1, cv2.LINE_AA)
            
            # Horizontal lines
            pt1 = np.array([[0, i * step]], dtype=np.float32)
            pt2 = np.array([[800, i * step]], dtype=np.float32)
            
            pt1_t = cv2.perspectiveTransform(pt1.reshape(1, 1, 2), M)[0][0]
            pt2_t = cv2.perspectiveTransform(pt2.reshape(1, 1, 2), M)[0][0]
            
            cv2.line(frame, tuple(pt1_t.astype(int)), 
                    tuple(pt2_t.astype(int)), color, 1, cv2.LINE_AA)
    
    def _draw_stats(self, frame: np.ndarray, current_frame: int, 
                    total_frames: int, status_text: str, 
                    status_color: tuple, summary: Dict):
        """Draw statistics overlay"""
        
        # Semi-transparent background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Frame info
        cv2.putText(frame, f"Frame: {current_frame}/{total_frames}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        progress = (current_frame / total_frames) * 100
        cv2.putText(frame, f"Progress: {progress:.1f}%", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detection status
        cv2.putText(frame, f"Status: {status_text}", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Overall detection rate
        rate = summary['detection_rate'] * 100
        cv2.putText(frame, f"Detection Rate: {rate:.1f}%", 
                   (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def create_side_by_side(self, output_path: str):
        """Create side-by-side comparison (original | with overlay)"""
        
        if not self.load_json():
            return
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"✗ Failed to open video")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output is double width
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        print(f"\nCreating side-by-side comparison...")
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            original = frame.copy()
            annotated = frame.copy()
            
            # Add detection overlay to right side
            if frame_idx < len(self.detection_data['frames']):
                frame_data = self.detection_data['frames'][frame_idx]
                
                if frame_data['detected'] and frame_data['corners']:
                    corners = np.array(frame_data['corners'], dtype=np.float32)
                    
                    cv2.polylines(annotated, [corners.astype(np.int32)], 
                                True, (0, 255, 0), 3)
                    
                    # Draw corners
                    for corner in corners:
                        cv2.circle(annotated, tuple(corner.astype(int)), 
                                 8, (0, 0, 255), -1)
                    
                    self._draw_grid(annotated, corners)
            
            # Add labels
            cv2.putText(original, "ORIGINAL", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated, "DETECTED BOARD", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Combine frames
            combined = np.hstack([original, annotated])
            out.write(combined)
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed: {frame_idx} frames")
        
        cap.release()
        out.release()
        
        print(f"✓ Side-by-side video saved: {output_path}\n")
    
    def extract_warped_boards(self, output_dir: str, 
                             sample_rate: int = 30):
        """Extract warped top-down view of detected boards"""
        
        if not self.load_json():
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return
        
        print(f"\nExtracting warped board views...")
        print(f"Sample rate: every {sample_rate} frames")
        print(f"Output directory: {output_path}")
        
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save every Nth frame
            if frame_idx % sample_rate == 0:
                if frame_idx < len(self.detection_data['frames']):
                    frame_data = self.detection_data['frames'][frame_idx]
                    
                    if frame_data['detected'] and frame_data['corners']:
                        corners = np.array(frame_data['corners'], dtype=np.float32)
                        
                        # Warp to top-down view
                        warped = self._warp_board(frame, corners)
                        
                        # Save
                        output_file = output_path / f"board_frame_{frame_idx:05d}.jpg"
                        cv2.imwrite(str(output_file), warped)
                        saved_count += 1
            
            frame_idx += 1
        
        cap.release()
        
        print(f"✓ Extracted {saved_count} warped board images\n")
    
    def _warp_board(self, frame: np.ndarray, corners: np.ndarray, 
                    size: int = 800) -> np.ndarray:
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


# CLI Tool
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Add board detection overlay to video')
    parser.add_argument('video', help='Input video file')
    parser.add_argument('json', help='Detection JSON file')
    parser.add_argument('-o', '--output', default='output_visualization.mp4',
                       help='Output video file')
    parser.add_argument('--mode', choices=['overlay', 'sidebyside', 'extract'], 
                       default='overlay',
                       help='Visualization mode')
    parser.add_argument('--no-corners', action='store_true',
                       help='Hide corner markers')
    parser.add_argument('--no-grid', action='store_true',
                       help='Hide grid overlay')
    parser.add_argument('--no-stats', action='store_true',
                       help='Hide statistics')
    parser.add_argument('--color', default='0,255,0',
                       help='Border color as R,G,B (default: 0,255,0)')
    parser.add_argument('--thickness', type=int, default=3,
                       help='Border thickness')
    parser.add_argument('--extract-dir', default='extracted_boards',
                       help='Directory for extracted boards')
    parser.add_argument('--sample-rate', type=int, default=30,
                       help='Sample rate for extraction (every N frames)')
    
    args = parser.parse_args()
    
    # Parse color
    color = tuple(map(int, args.color.split(',')))
    
    # Create tool
    tool = VideoOverlayTool(args.video, args.json)
    
    # Execute based on mode
    if args.mode == 'overlay':
        tool.create_visualization(
            args.output,
            show_corners=not args.no_corners,
            show_grid=not args.no_grid,
            show_stats=not args.no_stats,
            color=color,
            thickness=args.thickness
        )
    
    elif args.mode == 'sidebyside':
        tool.create_side_by_side(args.output)
    
    elif args.mode == 'extract':
        tool.extract_warped_boards(args.extract_dir, args.sample_rate)


if __name__ == "__main__":
    # Example usage without CLI
    print("="*60)
    print("VIDEO OVERLAY TOOL")
    print("="*60)
    
    # Basic usage
    tool = VideoOverlayTool(
        video_path="data/easy/1.mp4",
        json_path="data/easy/1_board_detection.json"
    )
    
    # Create visualization with overlay
    tool.create_visualization(
        output_path="11.mp4",
        show_corners=True,
        show_grid=True,
        show_stats=True,
        color=(0, 255, 0),  # Green
        thickness=3
    )
    
    # Or create side-by-side comparison
    # tool.create_side_by_side("output/comparison.mp4")
    
    # Or extract warped board images
    # tool.extract_warped_boards("output/boards", sample_rate=30)