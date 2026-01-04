"""Utility functions for card detection system"""
import cv2
import numpy as np
from typing import Tuple, List

def get_top_point(box):
    """Get the topmost point of a bounding box for label placement"""
    top_idx = np.argmin(box[:, 1])
    return box[top_idx]

def box_to_bbox(box: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert rotated box to axis-aligned bounding box (x, y, w, h)"""
    x_min = int(np.min(box[:, 0]))
    y_min = int(np.min(box[:, 1]))
    x_max = int(np.max(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def bbox_to_box(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Convert bounding box to corner points"""
    x, y, w, h = bbox
    return np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.int32)

def calculate_iou(bbox1: Tuple[int, int, int, int], 
                  bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union of two bounding boxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_overlap_ratio(bbox1: Tuple[int, int, int, int],
                            bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate what percentage of bbox1 overlaps with bbox2"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    
    return intersection / area1 if area1 > 0 else 0.0

def is_fully_inside_board(box: np.ndarray, board_corners: np.ndarray) -> bool:
    """Check if all corners of a box are inside board boundaries"""
    board_poly = board_corners.astype(np.int32)
    for corner in box:
        point = (float(corner[0]), float(corner[1]))
        if cv2.pointPolygonTest(board_poly, point, False) < 0:
            return False
    return True

def is_bbox_fully_inside_board(bbox: Tuple[int, int, int, int], 
                                board_corners: np.ndarray) -> bool:
    """Check if bounding box is fully inside board"""
    x, y, w, h = bbox
    corners = [(float(x), float(y)), (float(x+w), float(y)), 
               (float(x+w), float(y+h)), (float(x), float(y+h))]
    board_poly = board_corners.astype(np.int32)
    for corner in corners:
        if cv2.pointPolygonTest(board_poly, corner, False) < 0:
            return False
    return True

def assign_team(center: Tuple[float, float], board_center_x: float) -> str:
    """Assign team based on card position relative to board center"""
    if board_center_x is None:
        return None
    return 'A' if center[0] < board_center_x else 'B'

def are_cards_adjacent_vertically(pos1: Tuple[float, float], 
                                   pos2: Tuple[float, float],
                                   distance_threshold: float = 120) -> bool:
    """Check if two cards are adjacent along the y-axis (vertical battle position)"""
    # Calculate distance
    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    # Check if they're close enough
    if dist > distance_threshold:
        return False
    
    # Check if they're more vertically aligned than horizontally
    x_diff = abs(pos1[0] - pos2[0])
    y_diff = abs(pos1[1] - pos2[1])
    
    # Cards should be more aligned vertically (y difference should be larger)
    return y_diff > x_diff * 0.8

def merge_overlapping_detections(detections: List, overlap_threshold: float = 0.1):
    """Merge detections that overlap more than threshold, keeping the one with lower ID"""
    if not detections:
        return []
    
    # Sort by card_id (lower IDs first)
    sorted_detections = sorted(detections, key=lambda x: x.card_id if x.card_id else float('inf'))
    
    merged = []
    used = set()
    
    for i, det1 in enumerate(sorted_detections):
        if i in used:
            continue
        
        bbox1 = box_to_bbox(det1.box)
        should_merge = False
        
        for j, det2 in enumerate(sorted_detections[i+1:], start=i+1):
            if j in used:
                continue
            
            bbox2 = box_to_bbox(det2.box)
            overlap = calculate_overlap_ratio(bbox2, bbox1)
            
            if overlap > overlap_threshold:
                # Mark det2 as used (we keep det1 with lower ID)
                used.add(j)
                should_merge = True
        
        if not should_merge or det1.card_id is not None:
            merged.append(det1)
        used.add(i)
    
    return merged