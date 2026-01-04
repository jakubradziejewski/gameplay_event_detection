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
    battle_min_area: int = 17000
    battle_max_area: int = 36000
    battle_min_aspect: float = 0.3
    battle_max_aspect: float = 1.5

@dataclass
class Card:
    box: np.ndarray
    center: Tuple[float, float]
    width: float
    height: float
    card_id: Optional[int] = None
    is_battle: bool = False
    battle_ids: Optional[Tuple[int, int]] = None
    team: Optional[str] = None  # 'A' or 'B'

@dataclass
class TrackedObject:
    """Represents a tracked card or battle"""
    object_id: int
    tracker: any
    bbox: Tuple[int, int, int, int]
    is_battle: bool
    battle_ids: Optional[Tuple[int, int]] = None
    frames_lost: int = 0
    last_center: Tuple[float, float] = None
    team: Optional[str] = None  # Team assignment for cards
    # Battle-specific tracking
    initial_bbox: Optional[Tuple[int, int, int, int]] = None
    initial_card_positions: Optional[Dict[int, Tuple[float, float]]] = None
    battle_start_frame: Optional[int] = None
    frames_both_missing: int = 0
    frames_one_missing: int = 0
    missing_card_id: Optional[int] = None

class CardDetector:
    def __init__(self, params=None, tracker_type='KCF'):
        self.params = params or DetectionParams()
        self.tracker_type = tracker_type
        self.next_id = 1
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.max_frames_lost = 30
        self.previous_card_positions: Dict[int, Tuple[float, float]] = {}
        
        # Battle management
        self.battle_end_threshold = 40
        self.battle_noise_threshold = 40
        self.current_frame = 0
        
        # Team tracking
        self.board_center_x = None  # Will be set based on board detection
        
    def _create_tracker(self):
        """Create a new OpenCV tracker"""
        if self.tracker_type == 'CSRT':
            return cv2.legacy.TrackerCSRT_create()
        else:
            return cv2.legacy.TrackerKCF_create()

    def _assign_team(self, center: Tuple[float, float]) -> str:
        """Assign team based on card position relative to board center"""
        if self.board_center_x is None:
            return None
        
        x = center[0]
        return 'A' if x < self.board_center_x else 'B'

    def detect_cards(self, frame, board_corners=None):
        self.current_frame += 1
        
        # Calculate board center if available
        if board_corners is not None and self.board_center_x is None:
            self.board_center_x = np.mean(board_corners[:, 0])
        
        # Update all existing trackers
        self._update_trackers(frame)
        
        # Detect new cards/battles
        detected_cards, detected_battles = self._detect_objects(frame)
        
        # Filter to board boundaries
        if board_corners is not None:
            detected_cards = self._filter_fully_inside_board(detected_cards, board_corners)
            detected_battles = self._filter_fully_inside_board(detected_battles, board_corners)
        
        # BLOCK detection in active battle areas
        detected_cards = self._filter_battle_zones(detected_cards)
        detected_battles = self._filter_battle_zones(detected_battles)
        
        # Match detections with tracked objects or create new trackers
        all_objects = self._match_and_track(frame, detected_cards, detected_battles, board_corners)
        
        # Process battle states (check for endings, handle occlusions)
        all_objects = self._process_battle_states(all_objects, board_corners)
        
        # Separate cards and battles for return
        cards = [obj for obj in all_objects if not obj.is_battle]
        battles = [obj for obj in all_objects if obj.is_battle]
        
        return cards, battles
    
    def get_team_scores(self, cards: List[Card], battles: List[Card]) -> Dict[str, int]:
        """Calculate current score for each team based on cards on board"""
        scores = {'A': 0, 'B': 0}
        
        # Count individual cards
        for card in cards:
            if card.team:
                scores[card.team] += 1
        
        # Count cards in battles (one for each team)
        for battle in battles:
            if battle.battle_ids:
                for card_id in battle.battle_ids:
                    tracked = self.tracked_objects.get(card_id)
                    if tracked and tracked.team:
                        scores[tracked.team] += 1
        
        return scores
    
    def _filter_battle_zones(self, objects: List[Card]) -> List[Card]:
        """Remove any detections that overlap with active battle zones"""
        active_battles = [obj for obj in self.tracked_objects.values() if obj.is_battle]
        
        if not active_battles:
            return objects
        
        filtered = []
        for obj in objects:
            obj_center = obj.center
            
            # Check if this object overlaps with any battle zone
            overlaps_battle = False
            for battle in active_battles:
                bx, by, bw, bh = battle.bbox
                battle_center = (bx + bw/2, by + bh/2)
                
                # Check if object center is within battle zone (with margin)
                dist = np.sqrt((obj_center[0] - battle_center[0])**2 + 
                              (obj_center[1] - battle_center[1])**2)
                
                # Battle zone radius is roughly half the battle bbox diagonal
                battle_radius = np.sqrt(bw**2 + bh**2) / 2
                
                if dist < battle_radius * 1.2:  # 20% margin
                    overlaps_battle = True
                    break
            
            if not overlaps_battle:
                filtered.append(obj)
        
        return filtered
    
    def _process_battle_states(self, all_objects: List[Card], board_corners) -> List[Card]:
        """Process battle states: check for endings, handle occlusions, restore cards"""
        battles_to_end = []
        cards_to_restore = []
        
        for battle in [obj for obj in all_objects if obj.is_battle]:
            if battle.battle_ids is None:
                continue
            
            tracked_battle = self.tracked_objects.get(battle.card_id)
            if tracked_battle is None:
                continue
            
            card1_id, card2_id = battle.battle_ids
            
            # Check which cards are currently detected
            detected_card_ids = {obj.card_id for obj in all_objects if not obj.is_battle}
            card1_present = card1_id in detected_card_ids
            card2_present = card2_id in detected_card_ids
            
            # Count missing cards
            if not card1_present and not card2_present:
                # Both missing - likely hand occlusion (NOISE)
                tracked_battle.frames_both_missing += 1
                tracked_battle.frames_one_missing = 0
                tracked_battle.missing_card_id = None
                
                if tracked_battle.frames_both_missing > self.battle_noise_threshold:
                    tracked_battle.frames_both_missing = 0
                
            elif not card1_present or not card2_present:
                # One card missing - potential battle end
                missing_id = card1_id if not card1_present else card2_id
                
                if tracked_battle.missing_card_id == missing_id:
                    tracked_battle.frames_one_missing += 1
                else:
                    tracked_battle.missing_card_id = missing_id
                    tracked_battle.frames_one_missing = 1
                
                tracked_battle.frames_both_missing = 0
                
                # Check if battle should end
                if tracked_battle.frames_one_missing >= self.battle_end_threshold:
                    winner_id = card2_id if missing_id == card1_id else card1_id
                    battles_to_end.append((battle.card_id, winner_id, missing_id))
            
            else:
                # Both cards present - battle continues normally
                tracked_battle.frames_both_missing = 0
                tracked_battle.frames_one_missing = 0
                tracked_battle.missing_card_id = None
        
        # End battles and restore winner cards
        for battle_id, winner_id, loser_id in battles_to_end:
            tracked_battle = self.tracked_objects.get(battle_id)
            if tracked_battle and tracked_battle.initial_card_positions:
                # Restore winner card at its initial position
                winner_pos = tracked_battle.initial_card_positions.get(winner_id)
                winner_tracked = self.tracked_objects.get(winner_id)
                winner_team = winner_tracked.team if winner_tracked else None
                
                if winner_pos:
                    bx, by, bw, bh = tracked_battle.initial_bbox
                    card_w = bw / 2
                    card_h = bh
                    
                    card_bbox = (
                        int(winner_pos[0] - card_w/2),
                        int(winner_pos[1] - card_h/2),
                        int(card_w),
                        int(card_h)
                    )
                    
                    tracker = self._create_tracker()
                    
                    self.tracked_objects[winner_id] = TrackedObject(
                        object_id=winner_id,
                        tracker=tracker,
                        bbox=card_bbox,
                        is_battle=False,
                        battle_ids=None,
                        frames_lost=0,
                        last_center=winner_pos,
                        team=winner_team
                    )
                    
                    cards_to_restore.append((winner_id, card_bbox, winner_pos, winner_team))
            
            # Remove battle tracker
            if battle_id in self.tracked_objects:
                del self.tracked_objects[battle_id]
            
            all_objects = [obj for obj in all_objects if obj.card_id != battle_id]
        
        # Add restored cards back to results
        for winner_id, card_bbox, winner_pos, winner_team in cards_to_restore:
            x, y, w, h = card_bbox
            box = self._bbox_to_box(card_bbox)
            
            restored_card = Card(
                box=box,
                center=winner_pos,
                width=w,
                height=h,
                card_id=winner_id,
                is_battle=False,
                team=winner_team
            )
            all_objects.append(restored_card)
        
        return all_objects
    
    def _filter_fully_inside_board(self, objects: List[Card], board_corners: np.ndarray) -> List[Card]:
        """Filter objects to only include those FULLY inside board boundaries"""
        board_poly = board_corners.astype(np.int32)
        fully_inside = []
        
        for obj in objects:
            all_corners_inside = True
            for corner in obj.box:
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
            max_lost = self.battle_noise_threshold if tracked_obj.is_battle else self.max_frames_lost
            
            success, bbox = tracked_obj.tracker.update(frame)
            
            if success:
                tracked_obj.bbox = tuple(map(int, bbox))
                tracked_obj.frames_lost = 0
                x, y, w, h = tracked_obj.bbox
                tracked_obj.last_center = (x + w/2, y + h/2)
            else:
                tracked_obj.frames_lost += 1
                
                if not tracked_obj.is_battle and tracked_obj.frames_lost > max_lost:
                    to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
            if obj_id in self.previous_card_positions:
                del self.previous_card_positions[obj_id]
    
    def _detect_objects(self, frame):
        """Detect cards and battles using the original detection algorithm"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel_grad)
        
        _, grad_bin = cv2.threshold(gradient, self.params.gradient_threshold, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(grad_bin, self.params.canny_low, self.params.canny_high)
        
        kernel_conn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel_conn, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for c in contours:
            if cv2.contourArea(c) > 1000:
                cv2.drawContours(mask, [c], -1, 255, -1)

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, foreground = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
        foreground = np.uint8(foreground)
        
        background = cv2.dilate(mask, kernel_conn, iterations=3)
        unknown = cv2.subtract(background, foreground)
        
        ret, markers = cv2.connectedComponents(foreground)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(frame, markers)
        
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
        
        for detection in all_detections:
            best_match_id = None
            best_iou = 0.3
            
            detection_bbox = self._box_to_bbox(detection.box)
            
            for obj_id, tracked_obj in self.tracked_objects.items():
                if obj_id in matched_ids:
                    continue
                
                if tracked_obj.is_battle != detection.is_battle:
                    continue
                
                iou = self._calculate_iou(detection_bbox, tracked_obj.bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = obj_id
            
            if best_match_id is not None:
                tracked_obj = self.tracked_objects[best_match_id]
                tracked_obj.tracker = self._create_tracker()
                tracked_obj.tracker.init(frame, detection_bbox)
                tracked_obj.bbox = detection_bbox
                tracked_obj.frames_lost = 0
                
                detection.card_id = best_match_id
                detection.battle_ids = tracked_obj.battle_ids
                detection.team = tracked_obj.team
                result_objects.append(detection)
                matched_ids.add(best_match_id)
            else:
                if board_corners is None or self._is_fully_inside_board(detection, board_corners):
                    new_id = self.next_id
                    self.next_id += 1
                    
                    detection_bbox = self._box_to_bbox(detection.box)
                    tracker = self._create_tracker()
                    tracker.init(frame, detection_bbox)
                    
                    # Assign team for new cards
                    team = None
                    if not detection.is_battle:
                        team = self._assign_team(detection.center)
                    
                    battle_ids = None
                    initial_positions = None
                    
                    if detection.is_battle:
                        battle_ids = self._identify_battle_cards_from_position(detection.center)
                        
                        if battle_ids:
                            initial_positions = {}
                            for card_id in battle_ids:
                                if card_id in self.previous_card_positions:
                                    initial_positions[card_id] = self.previous_card_positions[card_id]
                    
                    self.tracked_objects[new_id] = TrackedObject(
                        object_id=new_id,
                        tracker=tracker,
                        bbox=detection_bbox,
                        is_battle=detection.is_battle,
                        battle_ids=battle_ids,
                        frames_lost=0,
                        last_center=detection.center,
                        team=team,
                        initial_bbox=detection_bbox if detection.is_battle else None,
                        initial_card_positions=initial_positions,
                        battle_start_frame=self.current_frame if detection.is_battle else None
                    )
                    
                    detection.card_id = new_id
                    detection.battle_ids = battle_ids
                    detection.team = team
                    result_objects.append(detection)
        
        for obj_id, tracked_obj in self.tracked_objects.items():
            if obj_id not in matched_ids and tracked_obj.frames_lost == 0:
                if board_corners is None or self._is_bbox_fully_inside_board(tracked_obj.bbox, board_corners):
                    box = self._bbox_to_box(tracked_obj.bbox)
                    x, y, w, h = tracked_obj.bbox
                    
                    card = Card(
                        box=box,
                        center=(x + w/2, y + h/2),
                        width=w,
                        height=h,
                        card_id=obj_id,
                        is_battle=tracked_obj.is_battle,
                        battle_ids=tracked_obj.battle_ids,
                        team=tracked_obj.team
                    )
                    result_objects.append(card)
                else:
                    if not tracked_obj.is_battle:
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
        
        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.is_battle:
                continue
            
            if tracked_obj.last_center is None:
                continue
            
            dist = np.sqrt((tracked_obj.last_center[0] - battle_center[0])**2 + 
                          (tracked_obj.last_center[1] - battle_center[1])**2)
            
            if dist < 150:
                nearby_cards.append((obj_id, dist))
        
        if len(nearby_cards) >= 2:
            nearby_cards.sort(key=lambda x: x[1])
            return (nearby_cards[0][0], nearby_cards[1][0])
        
        return None