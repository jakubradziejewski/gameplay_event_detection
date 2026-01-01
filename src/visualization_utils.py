import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class VisualizationUtils:
    """Utilities for visualizing board detection and various image filters"""
    
    @staticmethod
    def draw_adaptive_rectangular_grid(frame: np.ndarray, corners: np.ndarray, 
                                       grid_cols: int = 8, grid_rows: int = 6,
                                       color: tuple = (0, 255, 255), 
                                       show_labels: bool = False,
                                       line_thickness: int = 1):
        """Draw rectangular grid with cells that adapt to perspective"""
        
        tl, tr, br, bl = corners
        
        # Calculate vectors for interpolation
        top_vec = tr - tl
        bottom_vec = br - bl
        left_vec = bl - tl
        right_vec = br - tr
        
        # Draw vertical lines
        for col in range(grid_cols + 1):
            t = col / grid_cols
            top_pt = tl + top_vec * t
            bottom_pt = bl + bottom_vec * t
            
            thickness = 2 if col == 0 or col == grid_cols else line_thickness
            cv2.line(frame, tuple(top_pt.astype(int)), 
                    tuple(bottom_pt.astype(int)), color, thickness, cv2.LINE_AA)
        
        # Draw horizontal lines
        for row in range(grid_rows + 1):
            t = row / grid_rows
            left_pt = tl + left_vec * t
            right_pt = tr + right_vec * t
            
            thickness = 2 if row == 0 or row == grid_rows else line_thickness
            cv2.line(frame, tuple(left_pt.astype(int)), 
                    tuple(right_pt.astype(int)), color, thickness, cv2.LINE_AA)
        
        # Draw cell labels (A1-H6 style)
        if show_labels:
            for row in range(grid_rows):
                for col in range(grid_cols):
                    t_x = (col + 0.5) / grid_cols
                    t_y = (row + 0.5) / grid_rows
                    
                    top_pt = tl + top_vec * t_x
                    bottom_pt = bl + bottom_vec * t_x
                    center = top_pt + (bottom_pt - top_pt) * t_y
                    
                    label = f"{chr(65 + col)}{row + 1}"
                    cv2.putText(frame, label, tuple(center.astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    @staticmethod
    def draw_detection_on_frame(frame: np.ndarray, corners: np.ndarray, 
                               frame_count: int, total_frames: int,
                               detected_count: int, show_labels: bool,
                               grid_cols: int, grid_rows: int) -> np.ndarray:
        """Draw detection results on a single frame"""
        
        vis_frame = frame.copy()
        
        # Draw inner board boundary
        cv2.polylines(vis_frame, [corners.astype(np.int32)], 
                    True, (0, 255, 0), 3)
        
        # Draw adaptive rectangular grid
        VisualizationUtils.draw_adaptive_rectangular_grid(
            vis_frame, corners, grid_cols, grid_rows, (0, 255, 255), show_labels
        )
        
        # Draw corners
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        for corner, label in zip(corners, corner_labels):
            cv2.circle(vis_frame, tuple(corner.astype(int)), 8, (0, 0, 255), -1)
            cv2.circle(vis_frame, tuple(corner.astype(int)), 10, (255, 255, 255), 2)
            cv2.putText(vis_frame, label, tuple((corner + 15).astype(int)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show progress
        progress = f"Frame: {frame_count}/{total_frames} ({detected_count}/{frame_count} detected)"
        cv2.putText(vis_frame, progress, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(vis_frame, f"BOARD ({grid_cols}x{grid_rows})", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame
    
    @staticmethod
    def save_detection_steps_visualization(detection_data: dict, frame: np.ndarray,
                                          output_path: str = "output/detection_steps.png"):
        """Save comprehensive step-by-step visualization of detection process"""
        
        gray = detection_data['gray']
        blurred = detection_data['blurred']
        mask = detection_data['mask']
        mask_closed = detection_data['mask_closed']
        mask_cleaned = detection_data['mask_cleaned']
        contours = detection_data['contours']
        largest = detection_data['largest']
        corners = detection_data['corners']
        threshold = detection_data['threshold']
        area_percent = detection_data['area_percent']
        
        h, w = gray.shape
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
        if largest is not None:
            cv2.drawContours(step8, [largest], -1, (0, 255, 0), 3)
        cv2.putText(step8, "8. Largest Contour", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(step8, f"Area: {area_percent:.1f}% of frame", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        steps.append(step8)
        
        # Step 9: Approximated quadrilateral
        step9 = frame.copy()
        if corners is not None:
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
        
        cv2.imwrite(output_path, final_viz)
        print(f"✓ Detection steps visualization saved: {output_path}")
    
    @staticmethod
    def apply_filters(frame: np.ndarray) -> dict:
        """Apply various filters to frame for analysis"""
        
        filters = {}
        
        # Original
        filters['original'] = frame.copy()
        
        # Grayscale
        filters['grayscale'] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # HSV Color Space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        filters['hsv_full'] = hsv
        filters['hsv_hue'] = hsv[:, :, 0]
        filters['hsv_saturation'] = hsv[:, :, 1]
        filters['hsv_value'] = hsv[:, :, 2]
        
        # High Contrast (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        filters['high_contrast'] = clahe.apply(gray)
        
        # Edge Detection (Canny)
        filters['edges_canny'] = cv2.Canny(gray, 50, 150)
        
        # Sobel Edge Detection (X and Y)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        filters['edges_sobel_x'] = np.uint8(np.absolute(sobel_x))
        filters['edges_sobel_y'] = np.uint8(np.absolute(sobel_y))
        filters['edges_sobel_combined'] = np.uint8(np.sqrt(sobel_x**2 + sobel_y**2))
        
        # Laplacian (edge detection)
        filters['edges_laplacian'] = cv2.Laplacian(gray, cv2.CV_64F)
        filters['edges_laplacian'] = np.uint8(np.absolute(filters['edges_laplacian']))
        
        # Adaptive Thresholding
        filters['adaptive_thresh_mean'] = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        filters['adaptive_thresh_gaussian'] = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Color Channel Separation
        b, g, r = cv2.split(frame)
        filters['channel_blue'] = b
        filters['channel_green'] = g
        filters['channel_red'] = r
        
        # Lab Color Space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        filters['lab_l'] = lab[:, :, 0]  # Lightness
        filters['lab_a'] = lab[:, :, 1]  # Green-Red
        filters['lab_b'] = lab[:, :, 2]  # Blue-Yellow
        
        # Bilateral Filter (edge-preserving smoothing)
        filters['bilateral'] = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Morphological Gradient
        kernel = np.ones((5, 5), np.uint8)
        filters['morph_gradient'] = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Top Hat and Black Hat
        filters['tophat'] = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        filters['blackhat'] = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        return filters
    
    @staticmethod
    def save_filter_comparison(frame: np.ndarray, 
                              output_path: str = "output/filter_analysis.png"):
        """Save comprehensive filter comparison visualization"""
        
        filters = VisualizationUtils.apply_filters(frame)
        
        # Select key filters to display
        display_filters = [
            ('original', 'Original Frame', True),
            ('grayscale', 'Grayscale', False),
            ('high_contrast', 'High Contrast (CLAHE)', False),
            ('hsv_hue', 'HSV - Hue', False),
            ('hsv_saturation', 'HSV - Saturation', False),
            ('hsv_value', 'HSV - Value', False),
            ('edges_canny', 'Canny Edges', False),
            ('edges_sobel_combined', 'Sobel Edges', False),
            ('channel_red', 'Red Channel', False),
            ('channel_green', 'Green Channel', False),
            ('channel_blue', 'Blue Channel', False),
            ('lab_l', 'LAB - Lightness', False),
            ('adaptive_thresh_mean', 'Adaptive Threshold (Mean)', False),
            ('adaptive_thresh_gaussian', 'Adaptive Threshold (Gaussian)', False),
            ('morph_gradient', 'Morphological Gradient', False),
            ('tophat', 'Top Hat Transform', False),
        ]
        
        # Create grid of images (4x4)
        rows = []
        for i in range(0, len(display_filters), 4):
            row_filters = display_filters[i:i+4]
            row_images = []
            
            for filter_name, label, is_color in row_filters:
                img = filters[filter_name].copy()
                
                # Convert to BGR for display
                if not is_color and len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif not is_color and len(img.shape) == 3 and img.shape[2] == 3:
                    pass  # Already BGR
                
                # Add label
                cv2.putText(img, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add statistics
                if filter_name != 'original':
                    gray_img = filters[filter_name]
                    if len(gray_img.shape) == 3:
                        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
                    
                    avg_val = np.mean(gray_img)
                    std_val = np.std(gray_img)
                    cv2.putText(img, f"Avg: {avg_val:.1f} Std: {std_val:.1f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                row_images.append(img)
            
            # Pad if needed
            while len(row_images) < 4:
                row_images.append(np.zeros_like(row_images[0]))
            
            rows.append(np.hstack(row_images))
        
        # Stack all rows
        final_viz = np.vstack(rows)
        
        # Add title banner
        banner_height = 60
        banner = np.zeros((banner_height, final_viz.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, "IMAGE FILTER ANALYSIS - FIRST FRAME", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        final_viz = np.vstack([banner, final_viz])
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(output_path, final_viz)
        print(f"✓ Filter analysis saved: {output_path}")
        print(f"  Showing {len(display_filters)} different filters")