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
                                      debug: bool = False,
                                      save_steps_viz: bool = False,
                                      output_path: str = "output/detection_steps.png") -> Optional[np.ndarray]:
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
        
        # Save comprehensive step-by-step visualization
        if save_steps_viz and not self.calibration_visualized:
            self._save_detection_steps_visualization(
                frame, gray, blurred, mask, mask_closed, mask_cleaned,
                contours, largest, ordered_corners, threshold, area_percent, output_path
            )
            self.calibration_visualized = True
        
        return ordered_corners
    
    def _save_detection_steps_visualization(self, frame, gray, blurred, mask, 
                                           mask_closed, mask_cleaned, contours, 
                                           largest, corners, threshold, area_percent,
                                           output_path):
        """Save a comprehensive visualization of all detection steps"""
        
        h, w = gray.shape
        
        # Create visualizations for each step
        steps = []
        
        # Step 1: Original frame
        step1 = frame.copy()
        cv2.putText(step1, "1. Original Frame", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        steps.append(step1)
        
        # Step 2: Grayscale
        step2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        avg_brightness = np.mean(gray)
        max_brightness = np.max(gray)
        cv2.putText(step2, "2. Grayscale", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(step2, f"Avg: {avg_brightness:.0f}, Max: {max_brightness:.0f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        steps.append(step2)
        
        # Step 3: Blurred
        step3 = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        cv2.putText(step3, "3. Gaussian Blur (5x5)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        steps.append(step3)
        
        # Step 4: Threshold mask
        step4 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        bright_pixels = np.sum(mask > 0)
        bright_percent = (bright_pixels / (h * w)) * 100
        cv2.putText(step4, f"4. Threshold (>={threshold})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(step4, f"Bright pixels: {bright_percent:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        steps.append(step4)
        
        # Step 5: After morphological closing
        step5 = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR)
        cv2.putText(step5, "5. Morph Close (7x7, x3)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(step5, "Fills holes", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        steps.append(step5)
        
        # Step 6: After morphological opening
        step6 = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)
        cv2.putText(step6, "6. Morph Open (7x7, x2)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(step6, "Removes noise", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        steps.append(step6)
        
        # Step 7: All contours
        step7 = cv2.cvtColor(mask_cleaned.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(step7, contours, -1, (0, 255, 255), 2)
        cv2.putText(step7, f"7. Contours ({len(contours)} found)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        steps.append(step7)
        
        # Step 8: Largest contour only
        step8 = frame.copy()
        cv2.drawContours(step8, [largest], -1, (0, 255, 0), 3)
        cv2.putText(step8, "8. Largest Contour", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(step8, f"Area: {area_percent:.1f}% of frame", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        steps.append(step8)
        
        # Step 9: Approximated quadrilateral
        step9 = frame.copy()
        cv2.polylines(step9, [corners.astype(np.int32)], True, (0, 0, 255), 3)
        for i, corner in enumerate(corners):
            cv2.circle(step9, tuple(corner.astype(int)), 8, (255, 0, 0), -1)
            label = ['TL', 'TR', 'BR', 'BL'][i]
            cv2.putText(step9, label, tuple((corner + 15).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(step9, "9. Final Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(step9, "Ordered corners", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        steps.append(step9)
        
        # Arrange in 3x3 grid
        row1 = np.hstack(steps[0:3])
        row2 = np.hstack(steps[3:6])
        row3 = np.hstack(steps[6:9])
        
        final_viz = np.vstack([row1, row2, row3])
        
        # Add title banner
        banner_height = 80
        banner = np.zeros((banner_height, final_viz.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, "BOARD DETECTION - STEP BY STEP ANALYSIS", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(banner, f"Threshold: {threshold} | Detection: {'SUCCESS' if corners is not None else 'FAILED'}", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        final_viz = np.vstack([banner, final_viz])
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as PNG for better quality
        cv2.imwrite(output_path, final_viz)
        print(f"✓ Detection steps visualization saved: {output_path}")
        print(f"  Image shows all 9 steps of the detection process")
    
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
    
    def visualize_brightness_mask(self, frame: np.ndarray, threshold: int = 200,
                                  output_path: str = "output/brightness_mask_debug.jpg"):
        """Visualize the brightness mask for debugging"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Raw threshold
        _, mask_raw = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
        # Cleaned mask
        kernel = np.ones((7, 7), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours on cleaned mask
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        contour_vis = cv2.cvtColor(mask_cleaned.copy(), cv2.COLOR_GRAY2BGR)
        if contours:
            # Draw all contours in different colors
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Largest in green
                cv2.drawContours(contour_vis, [contour], -1, color, 3)
                
                # Add area percentage text
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    area_percent = area / (frame.shape[0] * frame.shape[1]) * 100
                    cv2.putText(contour_vis, f"{area_percent:.1f}%", (cx, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Create visualization with 4 panels
        vis_row1 = np.hstack([
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        ])
        
        vis_row2 = np.hstack([
            cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2BGR),
            contour_vis
        ])
        
        vis = np.vstack([vis_row1, vis_row2])
        
        # Add labels
        h = gray.shape[0]
        w = gray.shape[1]
        cv2.putText(vis, "1. Original Grayscale", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, "2. Gaussian Blur", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"3. Threshold (>={threshold})", (10, h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, "4. Cleaned + Contours", (w + 10, h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add threshold recommendation
        avg_brightness = np.mean(gray)
        max_brightness = np.max(gray)
        cv2.putText(vis, f"Avg brightness: {avg_brightness:.0f}, Max: {max_brightness:.0f}", 
                   (10, vis.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if max_brightness < threshold:
            cv2.putText(vis, f"WARNING: Max brightness < threshold! Try threshold={int(avg_brightness * 0.8)}", 
                       (10, vis.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(output_path, vis)
        print(f"✓ Brightness mask visualization saved: {output_path}")
        print(f"  Average brightness: {avg_brightness:.0f}")
        print(f"  Max brightness: {max_brightness:.0f}")
        print(f"  Current threshold: {threshold}")
        
        if max_brightness < threshold:
            recommended = int(avg_brightness * 0.8)
            print(f"  ⚠️ RECOMMENDATION: Lower threshold to ~{recommended}")
        
        return avg_brightness, max_brightness
    
    def _draw_adaptive_rectangular_grid(self, frame: np.ndarray, corners: np.ndarray, 
                                        color: tuple = (0, 255, 255), 
                                        show_labels: bool = False,
                                        line_thickness: int = 1):
        """Draw 8x6 grid with rectangular cells that adapt to perspective"""
        
        tl, tr, br, bl = corners
        
        # Calculate vectors for interpolation
        top_vec = tr - tl
        bottom_vec = br - bl
        left_vec = bl - tl
        right_vec = br - tr
        
        # Draw vertical lines (9 lines for 8 columns)
        for col in range(self.grid_cols + 1):
            t = col / self.grid_cols
            top_pt = tl + top_vec * t
            bottom_pt = bl + bottom_vec * t
            
            thickness = 2 if col == 0 or col == self.grid_cols else line_thickness
            cv2.line(frame, tuple(top_pt.astype(int)), 
                    tuple(bottom_pt.astype(int)), color, thickness, cv2.LINE_AA)
        
        # Draw horizontal lines (7 lines for 6 rows)
        for row in range(self.grid_rows + 1):
            t = row / self.grid_rows
            left_pt = tl + left_vec * t
            right_pt = tr + right_vec * t
            
            thickness = 2 if row == 0 or row == self.grid_rows else line_thickness
            cv2.line(frame, tuple(left_pt.astype(int)), 
                    tuple(right_pt.astype(int)), color, thickness, cv2.LINE_AA)
        
        # Draw cell labels (A1-H6 style)
        if show_labels:
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    t_x = (col + 0.5) / self.grid_cols
                    t_y = (row + 0.5) / self.grid_rows
                    
                    top_pt = tl + top_vec * t_x
                    bottom_pt = bl + bottom_vec * t_x
                    center = top_pt + (bottom_pt - top_pt) * t_y
                    
                    label = f"{chr(65 + col)}{row + 1}"
                    cv2.putText(frame, label, tuple(center.astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    def process_video(self, output_path: str = None, visualize: bool = True,
                     show_cell_labels: bool = False, 
                     brightness_threshold: int = 200,
                     save_brightness_debug: bool = True,
                     save_detection_steps: bool = True):
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
        
        # Save brightness debug for first frame
        if save_brightness_debug:
            print("\nAnalyzing brightness...")
            avg_bright, max_bright = self.visualize_brightness_mask(
                first_frame, 
                threshold=brightness_threshold,
                output_path="output/brightness_mask_debug.jpg"
            )
            print()
        
        print("Detecting inner board...")
        inner = self._detect_inner_from_brightness(
            first_frame, 
            brightness_threshold, 
            debug=True,
            save_steps_viz=save_detection_steps,
            output_path="output/detection_steps.png"
        )
        
        if inner is None:
            print("\n✗ Failed to detect inner board")
            print("\nTROUBLESHOOTING:")
            print("1. Check 'output/detection_steps.png' to see each detection step")
            print("2. Check 'output/brightness_mask_debug.jpg' for overall analysis")
            print("3. If the board is darker, try lowering brightness_threshold (e.g., 150 or 120)")
            print("4. If the board is brighter, try increasing it (e.g., 220)")
            print("5. Make sure the playable board area is significantly brighter than surroundings")
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
            
            # Detect inner board (no visualization after first frame)
            inner_corners = self._detect_inner_from_brightness(
                frame, 
                brightness_threshold,
                save_steps_viz=False
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
            if visualize and inner_corners is not None:
                vis_frame = frame.copy()
                
                # Draw inner board boundary
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
                
                cv2.putText(vis_frame, f"BOARD ({self.grid_cols}x{self.grid_rows})", (10, 60),
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
                    'detection_method': 'brightness_mask',
                    'brightness_threshold': brightness_threshold
                },
                'frames': results
            }, f, indent=2)
        
        print(f"✓ Results saved: {results_path}")
        return results


if __name__ == "__main__":
    detector = BoardDetector("data/easy/game_video.mp4")
    
    results = detector.process_video(
        output_path="output/board_detection_easy.mp4",
        visualize=True,
        show_cell_labels=False,
        brightness_threshold=180,
        save_brightness_debug=True,
        save_detection_steps=True  # NEW: Saves detailed step-by-step PNG
    )