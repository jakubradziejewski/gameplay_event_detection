import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class DetectionParams:
    gradient_threshold: int = 30
    canny_low: int = 50
    canny_high: int = 150
    min_area: int = 8500
    max_area: int = 18000
    min_aspect: float = 0.6
    max_aspect: float = 3.0
    solidity_threshold: float = 0.8

class CardDetector:
    def __init__(self, params=None):
        self.params = params or DetectionParams()

    def detect_cards(self, frame):
        # 1. Pre-processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Morphological Gradient
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel_grad)
        
        # 3. Threshold & Canny
        _, grad_bin = cv2.threshold(gradient, self.params.gradient_threshold, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(grad_bin, self.params.canny_low, self.params.canny_high)
        
        # 4. Create Filled Mask
        kernel_conn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel_conn, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for c in contours:
            if cv2.contourArea(c) > 1000:
                cv2.drawContours(mask, [c], -1, 255, -1)

        # 5. Watershed Separation
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, foreground = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
        foreground = np.uint8(foreground)
        
        background = cv2.dilate(mask, kernel_conn, iterations=3)
        unknown = cv2.subtract(background, foreground)
        
        ret, markers = cv2.connectedComponents(foreground)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(frame, markers)
        
        # 6. Final Extraction
        valid_boxes = []
        for label in np.unique(markers):
            if label <= 1: continue
            
            target_mask = np.zeros_like(gray)
            target_mask[markers == label] = 255
            cnts, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                c = cnts[0]
                area = cv2.contourArea(c)
                rect = cv2.minAreaRect(c)
                w, h = rect[1]
                
                if w == 0 or h == 0: continue
                aspect = max(w, h) / min(w, h)
                
                if (self.params.min_area * 0.5 <= area <= self.params.max_area and 
                    self.params.min_aspect <= aspect <= self.params.max_aspect):
                    box = cv2.boxPoints(rect)
                    valid_boxes.append(np.int32(box))
                    
        return valid_boxes