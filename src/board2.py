#!/usr/bin/env python3

"""
Simple edge-focused detection - similar to Sudoku approach
Focus on getting clean edges first, then find cells
"""

import cv2
import numpy as np
from pathlib import Path


class SimpleEdgeDetector:
    """Detect board and cells using adaptive threshold + edges"""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

    def load_first_frame(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.video_path}")
            return None
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def detect_edges_adaptive(self, frame: np.ndarray):
        """Detect edges using adaptive threshold (like Sudoku)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold - finds local edges
        adaptive = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or ADAPTIVE_THRESH_MEAN_C
            cv2.THRESH_BINARY, 
            11,  # block size
            2    # constant subtracted
        )
        
        # Canny on adaptive threshold
        edges = cv2.Canny(adaptive, 50, 150, apertureSize=3)
        
        return gray, adaptive, edges

    def detect_edges_gradient(self, frame: np.ndarray):
        """Detect edges using morphological gradient"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur first
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Morphological gradient - highlights edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold the gradient
        _, gradient_binary = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
        
        # Canny on gradient
        edges = cv2.Canny(gradient_binary, 50, 150, apertureSize=3)
        
        return gray, gradient, gradient_binary, edges

    def find_lines(self, edges: np.ndarray):
        """Find lines using Hough transform (like your Sudoku code)"""
        # Standard Hough (returns rho, theta)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        return lines

    def find_lines_p(self, edges: np.ndarray):
        """Find line segments using Probabilistic Hough"""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=80,
            minLineLength=100,
            maxLineGap=10
        )
        return lines

    def draw_hough_lines(self, frame: np.ndarray, lines):
        """Draw Hough lines (like your Sudoku code)"""
        result = frame.copy()
        if lines is None:
            return result
            
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return result

    def draw_hough_lines_p(self, frame: np.ndarray, lines):
        """Draw Probabilistic Hough line segments"""
        result = frame.copy()
        if lines is None:
            return result
            
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return result

    def detect_cells_from_edges(self, edges: np.ndarray, min_area: int = 500):
        """Find cells from edge image using contours"""
        # Close gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        
        cells = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Get bounding box
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            width, height = rect[1]
            if width == 0 or height == 0:
                continue
            
            # Must be reasonably square/rectangular
            aspect = max(width, height) / min(width, height)
            if aspect > 3.0:
                continue
            
            # Get center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            cells.append({
                'contour': contour,
                'box': box,
                'center': (cx, cy),
                'area': area,
                'aspect': aspect
            })
        
        return cells

    def process_adaptive(self, output_dir: str = "output_adaptive"):
        """Process using adaptive threshold approach"""
        print("\n" + "="*60)
        print("ADAPTIVE THRESHOLD APPROACH (like Sudoku)")
        print("="*60)
        
        frame = self.load_first_frame()
        if frame is None:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get edges
        gray, adaptive, edges = self.detect_edges_adaptive(frame)
        
        print(f"[1] Adaptive threshold + Canny edges")
        cv2.imwrite(str(output_path / "1_gray.png"), gray)
        cv2.imwrite(str(output_path / "2_adaptive.png"), adaptive)
        cv2.imwrite(str(output_path / "3_edges.png"), edges)
        
        # Find lines (standard Hough)
        print(f"[2] Finding lines with Hough transform")
        lines = self.find_lines(edges)
        if lines is not None:
            print(f"    Found {len(lines)} lines")
            vis_lines = self.draw_hough_lines(frame, lines)
            cv2.imwrite(str(output_path / "4_hough_lines.png"), vis_lines)
        
        # Find line segments (Probabilistic Hough)
        lines_p = self.find_lines_p(edges)
        if lines_p is not None:
            print(f"    Found {len(lines_p)} line segments")
            vis_lines_p = self.draw_hough_lines_p(frame, lines_p)
            cv2.imwrite(str(output_path / "5_hough_lines_p.png"), vis_lines_p)
        
        # Detect cells
        print(f"[3] Detecting cells from edges")
        cells = self.detect_cells_from_edges(edges)
        print(f"    Found {len(cells)} potential cells")
        
        # Visualize cells
        vis_cells = frame.copy()
        for cell in cells:
            cv2.drawContours(vis_cells, [cell['box']], -1, (0, 255, 0), 2)
            cv2.circle(vis_cells, cell['center'], 3, (0, 0, 255), -1)
        cv2.imwrite(str(output_path / "6_detected_cells.png"), vis_cells)
        
        print(f"\n✓ Saved results to {output_path}")

    def process_gradient(self, output_dir: str = "output_gradient"):
        """Process using morphological gradient approach"""
        print("\n" + "="*60)
        print("MORPHOLOGICAL GRADIENT APPROACH")
        print("="*60)
        
        frame = self.load_first_frame()
        if frame is None:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get edges
        gray, gradient, gradient_binary, edges = self.detect_edges_gradient(frame)
        
        print(f"[1] Morphological gradient + Canny edges")
        cv2.imwrite(str(output_path / "1_gray.png"), gray)
        cv2.imwrite(str(output_path / "2_gradient.png"), gradient)
        cv2.imwrite(str(output_path / "3_gradient_binary.png"), gradient_binary)
        cv2.imwrite(str(output_path / "4_edges.png"), edges)
        
        # Find lines
        print(f"[2] Finding lines with Hough transform")
        lines = self.find_lines(edges)
        if lines is not None:
            print(f"    Found {len(lines)} lines")
            vis_lines = self.draw_hough_lines(frame, lines)
            cv2.imwrite(str(output_path / "5_hough_lines.png"), vis_lines)
        
        # Find line segments
        lines_p = self.find_lines_p(edges)
        if lines_p is not None:
            print(f"    Found {len(lines_p)} line segments")
            vis_lines_p = self.draw_hough_lines_p(frame, lines_p)
            cv2.imwrite(str(output_path / "6_hough_lines_p.png"), vis_lines_p)
        
        # Detect cells
        print(f"[3] Detecting cells from edges")
        cells = self.detect_cells_from_edges(edges)
        print(f"    Found {len(cells)} potential cells")
        
        # Visualize cells
        vis_cells = frame.copy()
        for cell in cells:
            cv2.drawContours(vis_cells, [cell['box']], -1, (0, 255, 0), 2)
            cv2.circle(vis_cells, cell['center'], 3, (0, 0, 255), -1)
        cv2.imwrite(str(output_path / "7_detected_cells.png"), vis_cells)
        
        print(f"\n✓ Saved results to {output_path}")


def main():
    VIDEO_PATH = "data/hard/game_hard.mp4"
    
    if not Path(VIDEO_PATH).exists():
        print(f"File not found: {VIDEO_PATH}")
        return
    
    detector = SimpleEdgeDetector(VIDEO_PATH)
    
    # Try both approaches
    detector.process_adaptive()
    detector.process_gradient()


if __name__ == "__main__":
    main()