import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

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
    # Battle detection parameters
    battle_min_area: int = 17000  # ~2x card area
    battle_max_area: int = 36000  # ~2x card area
    battle_min_aspect: float = 0.3  # Wider aspect ratio (two cards side by side)
    battle_max_aspect: float = 1.5

@dataclass
class Card:
    box: np.ndarray
    center: Tuple[float, float]
    width: float
    height: float
    card_id: Optional[int] = None
    is_battle: bool = False
    battle_ids: Optional[Tuple[int, int]] = None  # For battles: (card1_id, card2_id)

@dataclass
class TrackedObject:
    """Represents a tracked card or battle"""
    object_id: int
    tracker: any  # OpenCV tracker
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    is_battle: bool
    battle_ids: Optional[Tuple[int, int]] = None
    frames_lost: int = 0
    last_center: Tuple[float, float] = None

class CardDetector:
    def __init__(self, params=None, tracker_type='KCF'):
        self.params = params or DetectionParams()
        self.tracker_type = tracker_type  # 'KCF' or 'CSRT'
        self.next_id = 1
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.max_frames_lost = 30  # Remove tracker after 30 frames of failure
        self.previous_card_positions: Dict[int, Tuple[float, float]] = {}  # Track movement
        
    def _create_tracker(self):
        """Create a new OpenCV tracker"""
        if self.tracker_type == 'CSRT':
            return cv2.legacy.TrackerCSRT_create()
        else:  # KCF (default)
            return cv2.legacy.TrackerKCF_create()

    def detect_cards(self, frame, board_corners=None):
        # First, update all existing trackers
        self._update_trackers(frame)
        
        # Then, detect new cards/battles
        detected_cards, detected_battles = self._detect_objects(frame)
        
        # Filter to board boundaries and check if fully inside
        if board_corners is not None:
            detected_cards = self._filter_fully_inside_board(detected_cards, board_corners)
            detected_battles = self._filter_fully_inside_board(detected_battles, board_corners)
        
        # Match detections with tracked objects or create new trackers
        all_objects = self._match_and_track(frame, detected_cards, detected_battles, board_corners)
        
        # Separate cards and battles for return
        cards = [obj for obj in all_objects if not obj.is_battle]
        battles = [obj for obj in all_objects if obj.is_battle]
        
        return cards, battles
    
    def _filter_fully_inside_board(self, objects: List[Card], board_corners: np.ndarray) -> List[Card]:
        """Filter objects to only include those FULLY inside board boundaries"""
        board_poly = board_corners.astype(np.int32)
        fully_inside = []
        
        for obj in objects:
            # Check if ALL corners of the object box are inside the board
            all_corners_inside = True
            for corner in obj.box:
                # Convert corner to proper tuple format (float, float)
                point = (float(corner[0]), float(corner[1]))
                if cv2.pointPolygonTest(board_poly, point, False) < 0:
                    all_corners_inside = False
                    break
            
            if all_corners_inside:
                fully_inside.append(obj)
        
        return fully_inside
    
    def _update_trackers(self, frame):
        """Update all existing trackers"""
        to_remove = []
        
        for obj_id, tracked_obj in self.tracked_objects.items():
            success, bbox = tracked_obj.tracker.update(frame)
            
            if success:
                tracked_obj.bbox = tuple(map(int, bbox))
                tracked_obj.frames_lost = 0
                # Update center
                x, y, w, h = tracked_obj.bbox
                tracked_obj.last_center = (x + w/2, y + h/2)
            else:
                tracked_obj.frames_lost += 1
                if tracked_obj.frames_lost > self.max_frames_lost:
                    to_remove.append(obj_id)
        
        # Remove lost trackers
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
            if obj_id in self.previous_card_positions:
                del self.previous_card_positions[obj_id]
    
    def _detect_objects(self, frame):
        """Detect cards and battles using the original detection algorithm"""
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
        
        # 6. Extract both single cards and battle pairs
        cards = []
        battles = []
        
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
                
                # Check if it's a battle (wider rectangle, larger area)
                if (self.params.battle_min_area <= area <= self.params.battle_max_area and 
                    self.params.battle_min_aspect <= aspect <= self.params.battle_max_aspect):
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    center = rect[0]
                    
                    battle = Card(
                        box=box,
                        center=center,
                        width=w,
                        height=h,
                        is_battle=True
                    )
                    battles.append(battle)
                    
                # Check if it's a single card
                elif (self.params.min_area * 0.5 <= area <= self.params.max_area and 
                      self.params.min_aspect <= aspect <= self.params.max_aspect):
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    center = rect[0]
                    
                    card = Card(
                        box=box,
                        center=center,
                        width=w,
                        height=h,
                        is_battle=False
                    )
                    cards.append(card)
        
        # DON'T remove overlapping cards - battles are detected separately
        # Cards under battles won't be detected anyway due to watershed
        
        return cards, battles

    def _box_to_bbox(self, box: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert rotated box to axis-aligned bounding box (x, y, w, h)"""
        x_min = int(np.min(box[:, 0]))
        y_min = int(np.min(box[:, 1]))
        x_max = int(np.max(box[:, 0]))
        y_max = int(np.max(box[:, 1]))
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _bbox_to_box(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Convert bounding box to corner points"""
        x, y, w, h = bbox
        return np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.int32)
    
    def _match_and_track(self, frame, detected_cards: List[Card], detected_battles: List[Card], board_corners) -> List[Card]:
        """Match detected objects with tracked objects or create new trackers"""
        all_detections = detected_cards + detected_battles
        matched_ids = set()
        result_objects = []
        
        # Try to match each detection with existing trackers
        for detection in all_detections:
            best_match_id = None
            best_iou = 0.3  # Minimum IoU threshold for matching
            
            detection_bbox = self._box_to_bbox(detection.box)
            
            # Check against all tracked objects of the same type
            for obj_id, tracked_obj in self.tracked_objects.items():
                if obj_id in matched_ids:
                    continue
                
                if tracked_obj.is_battle != detection.is_battle:
                    continue
                
                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(detection_bbox, tracked_obj.bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = obj_id
            
            if best_match_id is not None:
                # Update existing tracker
                tracked_obj = self.tracked_objects[best_match_id]
                tracked_obj.tracker = self._create_tracker()
                tracked_obj.tracker.init(frame, detection_bbox)
                tracked_obj.bbox = detection_bbox
                tracked_obj.frames_lost = 0
                
                # Create Card object with existing ID
                detection.card_id = best_match_id
                detection.battle_ids = tracked_obj.battle_ids
                result_objects.append(detection)
                matched_ids.add(best_match_id)
            else:
                # Create new tracker ONLY if on board
                if board_corners is None or self._is_fully_inside_board(detection, board_corners):
                    new_id = self.next_id
                    self.next_id += 1
                    
                    detection_bbox = self._box_to_bbox(detection.box)
                    tracker = self._create_tracker()
                    tracker.init(frame, detection_bbox)
                    
                    # For battles, try to identify which cards are involved
                    battle_ids = None
                    if detection.is_battle:
                        battle_ids = self._identify_battle_cards_from_position(detection.center)
                    
                    self.tracked_objects[new_id] = TrackedObject(
                        object_id=new_id,
                        tracker=tracker,
                        bbox=detection_bbox,
                        is_battle=detection.is_battle,
                        battle_ids=battle_ids,
                        frames_lost=0,
                        last_center=detection.center
                    )
                    
                    detection.card_id = new_id
                    detection.battle_ids = battle_ids
                    result_objects.append(detection)
        
        # Add tracked objects that weren't matched (still being tracked)
        for obj_id, tracked_obj in self.tracked_objects.items():
            if obj_id not in matched_ids and tracked_obj.frames_lost == 0:
                # Check if tracked object is still on board
                if board_corners is None or self._is_bbox_fully_inside_board(tracked_obj.bbox, board_corners):
                    # Create Card object from tracker
                    box = self._bbox_to_box(tracked_obj.bbox)
                    x, y, w, h = tracked_obj.bbox
                    
                    card = Card(
                        box=box,
                        center=(x + w/2, y + h/2),
                        width=w,
                        height=h,
                        card_id=obj_id,
                        is_battle=tracked_obj.is_battle,
                        battle_ids=tracked_obj.battle_ids
                    )
                    result_objects.append(card)
                else:
                    # Object left the board, remove it
                    del self.tracked_objects[obj_id]
                    if obj_id in self.previous_card_positions:
                        del self.previous_card_positions[obj_id]
        
        return result_objects
    
    def _is_fully_inside_board(self, obj: Card, board_corners: np.ndarray) -> bool:
        """Check if object is fully inside board"""
        board_poly = board_corners.astype(np.int32)
        for corner in obj.box:
            point = (float(corner[0]), float(corner[1]))
            if cv2.pointPolygonTest(board_poly, point, False) < 0:
                return False
        return True
    
    def _is_bbox_fully_inside_board(self, bbox: Tuple[int, int, int, int], board_corners: np.ndarray) -> bool:
        """Check if bounding box is fully inside board"""
        x, y, w, h = bbox
        corners = [(float(x), float(y)), (float(x+w), float(y)), 
                   (float(x+w), float(y+h)), (float(x), float(y+h))]
        board_poly = board_corners.astype(np.int32)
        for corner in corners:
            if cv2.pointPolygonTest(board_poly, corner, False) < 0:
                return False
        return True
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
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
    
    def _identify_battle_cards_from_position(self, battle_center: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Identify which cards entered battle based on recent positions near battle location"""
        nearby_cards = []
        
        # Look at all tracked cards (not battles) near this position
        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.is_battle:
                continue
            
            if tracked_obj.last_center is None:
                continue
            
            # Calculate distance from tracked card to battle center
            dist = np.sqrt((tracked_obj.last_center[0] - battle_center[0])**2 + 
                          (tracked_obj.last_center[1] - battle_center[1])**2)
            
            if dist < 150:  # Within 150 pixels
                nearby_cards.append((obj_id, dist))
        
        if len(nearby_cards) >= 2:
            # Sort by distance and take closest two
            nearby_cards.sort(key=lambda x: x[1])
            return (nearby_cards[0][0], nearby_cards[1][0])
        
        return None