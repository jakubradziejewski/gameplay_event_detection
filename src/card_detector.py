import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from detection_steps_viz import save_detection_visualization

@dataclass
class DetectionParams:
    min_area: int = 10000
    max_area: int = 15000
    min_aspect: float = 1.3
    max_aspect: float = 2.4
    battle_proximity_buffer: float = 0.10

@dataclass
class Card:
    box: np.ndarray
    center: Tuple[float, float]
    width: float
    height: float
    card_id: Optional[int] = None
    team: Optional[str] = None

@dataclass
class Battle:
    battle_id: int
    card1_id: int
    card2_id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    box: np.ndarray  # 4 corners for drawing
    team1: str
    team2: str
    initial_center: Tuple[float, float] = None  # Track initial battle center
    last_valid_bbox: Tuple[int, int, int, int] = None  # Keep last known battle zone
    card1_in_zone: bool = True  # Track if card1 is still in battle zone
    card2_in_zone: bool = True  # Track if card2 is still in battle zone

@dataclass
class TrackedObject:
    object_id: int
    tracker: any
    bbox: Tuple[int, int, int, int]
    team: Optional[str] = None
    frames_lost: int = 0
    last_center: Tuple[float, float] = None
    last_valid_bbox: Optional[Tuple[int, int, int, int]] = None
    last_valid_box: Optional[np.ndarray] = None

class CardDetector:
    def __init__(self, params=None, enable_visualization=False, viz_output_dir="output_visualization"):
        self.params = params or DetectionParams()
        self.next_id = 1
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.previous_card_positions: Dict[int, Tuple[float, float]] = {}
        self.battles: Dict[int, Battle] = {}
        self.next_battle_id = 1
        self.current_frame = 0
        self.board_center_x = None
        self.edge_buffer = None
        
        # Visualization setup
        self.enable_visualization = enable_visualization
        self.viz_output_dir = Path(viz_output_dir)
        if enable_visualization:
            self.viz_output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_frame_number = 0

    def _create_tracker(self):
        #return cv2.legacy.TrackerCSRT_create()
        return cv2.legacy.TrackerKCF_create()

    def _assign_team(self, center: Tuple[float, float]) -> str:
        return 'A' if center[0] < self.board_center_x else 'B' if self.board_center_x else None
    
    def detect_cards(self, frame, board_corners=None):
        self.current_frame += 1
        if board_corners is not None:
            if self.board_center_x is None:
                self.board_center_x = np.mean(board_corners[:, 0])
        self._update_trackers(frame)
        detected_cards = self._detect_cards_only(frame)
        
        if board_corners is not None:
            detected_cards = self._filter_inside_board(detected_cards, board_corners)

        detected_cards = self._merge_overlaps(detected_cards)
        
        all_cards = self._match_and_track(frame, detected_cards, board_corners)
        
        # Detect battles from card positions
        self._detect_battles_from_cards(all_cards)
        
        # Process battles (check if cards left battle zones)
        battle_results = self._process_battles(all_cards)
        
        return all_cards, battle_results
    
    def _has_enough_features(self, gray_frame, box):
        # Crop to the card area
        x, y, w, h = self._box_to_bbox(box)
        roi = gray_frame[y:y+h, x:x+w]
        if roi.size == 0: return False
        
        orb = cv2.ORB_create(nfeatures=100)
        keypoints = orb.detect(roi, None)
        
        # A card should have dozens of keypoints; an empty cell will have very few
        return len(keypoints) > 15
    
    def _detect_cards_only(self, frame):
        self.viz_frame_number += 1
        save_viz = self.enable_visualization and (self.viz_frame_number % 200 == 0)
        
        # Step 1: Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Blur (9x9) - removes internal card details
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Step 3: Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Step 4: Dilation (3x3, 1 iteration)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                        (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Step 5: Fill mask from contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for c in contours:
            if cv2.contourArea(c) > 1000:
                cv2.drawContours(mask, [c], -1, 255, -1)
        
        # Step 6: Distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Step 7: Threshold (0.5) for confident centers
        _, fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        fg = np.uint8(fg)
        
        # Step 8: Background expansion (2 iterations)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bg = cv2.dilate(mask, kernel2, iterations=2)
        
        # Step 9: Unknown region
        unknown = cv2.subtract(bg, fg)
        
        # Watershed markers
        _, markers = cv2.connectedComponents(fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers_copy = markers.copy()
        markers = cv2.watershed(frame, markers)
        
        # Extract cards
        cards = []
        for label in np.unique(markers):
            if label <= 1:
                continue
            
            temp_mask = np.zeros_like(gray)
            temp_mask[markers == label] = 255
            
            cs, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            c = cs[0]
            
            area = cv2.contourArea(c)
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            
            box = np.int32(cv2.boxPoints(rect))
            x_coords = box[:, 0]
            y_coords = box[:, 1]
            actual_width = np.max(x_coords) - np.min(x_coords)
            actual_height = np.max(y_coords) - np.min(y_coords)

            # Reject if wider than tall
            if actual_width > actual_height:
                continue

            # Now normalize w, h for aspect ratio calculation
            if h < w:
                w, h = h, w
            aspect = h / w
            p = self.params
            if p.min_area <= area <= p.max_area and p.min_aspect <= aspect <= p.max_aspect:
                if self._has_enough_features(gray, box):
                    cards.append(Card(box=box, center=rect[0], width=w, height=h))
        
        # Save visualization if needed
        if save_viz:
            save_detection_visualization(
                self.viz_output_dir, self.viz_frame_number,
                frame, gray, blur, edges, dilated, mask, 
                dist, fg, bg, unknown, markers,
                cards
            )
        
        return cards
            
    def _detect_battles_from_cards(self, cards: List[Card]):
        """Detect battles by checking if opposing team cards are close to each other"""
        
        active_cards = [c for c in cards if c.team]
        
        # Group cards by team
        team_a = [c for c in active_cards if c.team == 'A']
        team_b = [c for c in active_cards if c.team == 'B']
        
        # Check each pair of opposing cards
        new_battles = []
        for card_a in team_a:
            for card_b in team_b:
                if self._are_cards_in_battle(card_a, card_b):
                    # Check if this battle already exists
                    battle_exists = False
                    for battle in self.battles.values():
                        if {battle.card1_id, battle.card2_id} == {card_a.card_id, card_b.card_id}:
                            battle_exists = True
                            # Update battle bbox to track card movement
                            battle.bbox = self._create_battle_bbox(card_a, card_b)
                            battle.box = self._bbox_to_box(battle.bbox)
                            break
                    
                    if not battle_exists:
                        # Create new battle
                        battle_bbox = self._create_battle_bbox(card_a, card_b)
                        battle_box = self._bbox_to_box(battle_bbox)
                        battle_center = ((card_a.center[0] + card_b.center[0]) / 2, 
                                       (card_a.center[1] + card_b.center[1]) / 2)
                        
                        battle = Battle(
                            battle_id=self.next_battle_id,
                            card1_id=card_a.card_id,
                            card2_id=card_b.card_id,
                            bbox=battle_bbox,
                            box=battle_box,
                            team1=card_a.team,
                            team2=card_b.team,
                            initial_center=battle_center,
                            last_valid_bbox=battle_bbox,
                            card1_in_zone=True,
                            card2_in_zone=True
                        )
                        self.battles[self.next_battle_id] = battle
                        self.next_battle_id += 1
                        new_battles.append(battle)
    
    def _are_cards_in_battle(self, card1: Card, card2: Card) -> bool:
        """Check if two cards are close enough to be in battle (20% buffer)"""
        bbox1 = self._box_to_bbox(card1.box)
        bbox2 = self._box_to_bbox(card2.box)
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate buffer (20% of card size)
        buffer1 = max(w1, h1) * self.params.battle_proximity_buffer
        buffer2 = max(w2, h2) * self.params.battle_proximity_buffer
        
        # Expand bboxes by buffer
        expanded1 = (x1 - buffer1, y1 - buffer1, w1 + 2*buffer1, h1 + 2*buffer1)
        expanded2 = (x2 - buffer2, y2 - buffer2, w2 + 2*buffer2, h2 + 2*buffer2)
        
        # Check if they intersect
        return self._bboxes_intersect(expanded1, expanded2)
    
    def _bboxes_intersect(self, bbox1: Tuple[float, float, float, float], 
                          bbox2: Tuple[float, float, float, float]) -> bool:
        """Check if two bounding boxes intersect"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    def _create_battle_bbox(self, card1: Card, card2: Card) -> Tuple[int, int, int, int]:
        """Create a bounding box that encompasses both cards"""
        bbox1 = self._box_to_bbox(card1.box)
        bbox2 = self._box_to_bbox(card2.box)
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Find the encompassing bbox
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2)
        y_max = max(y1 + h1, y2 + h2)
        
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    
    def _process_battles(self, cards: List[Card]) -> List[Battle]:
        """Check if battles have ended and track which card left the battle zone first"""
        battles_to_remove = []
        battle_events = []
        
        for battle_id, battle in list(self.battles.items()):
            # Find the two cards involved
            card1 = next((c for c in cards if c.card_id == battle.card1_id), None)
            card2 = next((c for c in cards if c.card_id == battle.card2_id), None)
            
            # Check if both cards still exist
            card1_exists = card1 is not None
            card2_exists = card2 is not None

            if card1_exists and card2_exists:
                # Store the current battle bbox as the last valid one
                battle.last_valid_bbox = battle.bbox
                
                # Update battle bbox to follow cards as they move
                battle.bbox = self._create_battle_bbox(card1, card2)
                battle.box = self._bbox_to_box(battle.bbox)
                
                # Check if each card is still in the ORIGINAL battle zone
                card1_still_in = self._is_card_in_battle_zone(card1, battle)
                card2_still_in = self._is_card_in_battle_zone(card2, battle)
                
                # Track who left the zone first
                if battle.card1_in_zone and not card1_still_in:
                    battles_to_remove.append(battle_id)
                    battle_events.append(f"Card {battle.card1_id} ({battle.team1}) lost the battle")
                elif battle.card2_in_zone and not card2_still_in:
                    battles_to_remove.append(battle_id)
                    battle_events.append(f"Card {battle.card2_id} ({battle.team2}) lost the battle")
                elif not self._are_cards_in_battle(card1, card2):
                    # Both cards moved too far apart
                    dist1 = np.sqrt((card1.center[0] - battle.initial_center[0])**2 + 
                                  (card1.center[1] - battle.initial_center[1])**2)
                    dist2 = np.sqrt((card2.center[0] - battle.initial_center[0])**2 + 
                                  (card2.center[1] - battle.initial_center[1])**2)
                    
                    if dist1 > dist2:
                        battles_to_remove.append(battle_id)
                        battle_events.append(f"Card {battle.card1_id} ({battle.team1}) lost the battle")
                    else:
                        battles_to_remove.append(battle_id)
                        battle_events.append(f"Card {battle.card2_id} ({battle.team2}) lost the battle")
                else:
                    # Update zone tracking
                    battle.card1_in_zone = card1_still_in
                    battle.card2_in_zone = card2_still_in
                    
            elif card1_exists and not card2_exists:
                battles_to_remove.append(battle_id)
                battle_events.append(f"Card {battle.card2_id} ({battle.team2}) lost the battle")
            elif card2_exists and not card1_exists:
                battles_to_remove.append(battle_id)
                battle_events.append(f"Card {battle.card1_id} ({battle.team1}) lost the battle")
            else:
                battles_to_remove.append(battle_id)
        
        # Remove ended battles
        for battle_id in battles_to_remove:
            del self.battles[battle_id]
        
        # Store events for display
        self.battle_events = battle_events
        
        # Return active battles as list
        return list(self.battles.values())
    
    def _is_card_in_battle_zone(self, card: Card, battle: Battle) -> bool:
        """Check if card center is within the last valid battle bounding box"""
        bbox_to_check = battle.last_valid_bbox if battle.last_valid_bbox else battle.bbox
        bx, by, bw, bh = bbox_to_check
        cx, cy = card.center
        
        return bx <= cx <= bx + bw and by <= cy <= by + bh
    
    def _merge_overlaps(self, dets):
        if not dets:
            return []
        tracked = sorted([d for d in dets if d.card_id], key=lambda x: x.card_id)
        merged, used = [], set()
        for i, d1 in enumerate(tracked + [d for d in dets if not d.card_id]):
            if i in used:
                continue
            b1 = self._box_to_bbox(d1.box)
            for j in range(i + 1, len(dets)):
                if j not in used and self._overlap_ratio(self._box_to_bbox(dets[j].box), b1) > 0.1:
                    used.add(j)
            merged.append(d1)
            used.add(i)
        return merged
    
    def _overlap_ratio(self, b1, b2):
        x1, y1, w1, h1, x2, y2, w2, h2 = *b1, *b2
        xi, yi = max(x1, x2), max(y1, y2)
        xf, yf = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        return max(0, (xf - xi) * (yf - yi)) / (w1 * h1) if xf > xi and yf > yi and w1 * h1 > 0 else 0
    
    def get_team_scores(self, cards, battles):
        scores = {'A': 0, 'B': 0}
        for c in cards:
            if c.team:
                scores[c.team] += 1
        return scores
    
    def _filter_inside_board(self, objs, corners):
        poly = corners.astype(np.int32)
        return [o for o in objs if all(cv2.pointPolygonTest(poly, (float(c[0]), float(c[1])), False) >= 0 for c in o.box)]
    
    def _update_trackers(self, frame):
        to_remove = []
        for oid, t in list(self.tracked_objects.items()):
            ok, bbox = t.tracker.update(frame)
            if ok:
                t.bbox = tuple(map(int, bbox))
                t.frames_lost = 0
                x, y, w, h = t.bbox
                t.last_center = (x+w/2, y+h/2)
                t.last_valid_bbox, t.last_valid_box = t.bbox, self._bbox_to_box(t.bbox)
            else:
                t.frames_lost += 1
                if t.frames_lost > 0:
                    to_remove.append(oid)
        for oid in to_remove:
            self.tracked_objects.pop(oid, None)
            self.previous_card_positions.pop(oid, None)

    def _box_to_bbox(self, box):
        return (int(np.min(box[:, 0])), int(np.min(box[:, 1])),
                int(np.max(box[:, 0]) - np.min(box[:, 0])), int(np.max(box[:, 1]) - np.min(box[:, 1])))
    
    def _bbox_to_box(self, bbox):
        x, y, w, h = bbox
        return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
    
    def _iou(self, b1, b2):
        x1, y1, w1, h1, x2, y2, w2, h2 = *b1, *b2
        xi, yi = max(x1, x2), max(y1, y2)
        xf, yf = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        if xf < xi or yf < yi:
            return 0
        inter = (xf-xi) * (yf-yi)
        return inter / (w1*h1 + w2*h2 - inter) if w1*h1 + w2*h2 > inter else 0
    
    def _match_and_track(self, frame, cards, board):
        matched, result = set(), []
        for det in cards:
            bbox = self._box_to_bbox(det.box)
            best_id, best_iou = None, 0.3
            for oid, t in self.tracked_objects.items():
                if oid in matched:
                    continue
                if (iou := self._iou(bbox, t.bbox)) > best_iou:
                    best_iou, best_id = iou, oid
            
            if best_id:
                t = self.tracked_objects[best_id]
                t.tracker = self._create_tracker()
                t.tracker.init(frame, bbox)
                t.bbox, t.frames_lost = bbox, 0
                t.last_valid_bbox, t.last_valid_box = bbox, det.box.copy()
                det.card_id, det.team = best_id, t.team
                result.append(det)
                matched.add(best_id)
            else:
                if any(self._overlap_ratio(bbox, o.bbox) > 0.1 for o in self.tracked_objects.values()):
                    continue
                if board is None or all(cv2.pointPolygonTest(board.astype(np.int32), (float(c[0]), float(c[1])), False) >= 0 
                                       for c in det.box):
                    nid = self.next_id
                    self.next_id += 1
                    tr = self._create_tracker()
                    tr.init(frame, bbox)
                    team = self._assign_team(det.center)
                    self.tracked_objects[nid] = TrackedObject(
                        object_id=nid, tracker=tr, bbox=bbox, team=team,
                        last_center=det.center, last_valid_bbox=bbox, last_valid_box=det.box.copy()
                    )
                    det.card_id, det.team = nid, team
                    result.append(det)
        
        for oid, t in list(self.tracked_objects.items()):
            if oid not in matched and t.frames_lost == 0:
                if board is None or all(cv2.pointPolygonTest(board.astype(np.int32), (float(x), float(y)), False) >= 0 
                                       for x, y in [(t.bbox[0], t.bbox[1]), (t.bbox[0]+t.bbox[2], t.bbox[1]),
                                                   (t.bbox[0]+t.bbox[2], t.bbox[1]+t.bbox[3]), (t.bbox[0], t.bbox[1]+t.bbox[3])]):
                    x, y, w, h = t.bbox
                    result.append(Card(box=self._bbox_to_box(t.bbox), center=(x+w/2, y+h/2),
                                     width=w, height=h, card_id=oid, team=t.team))
        
        for o in result:
            if o.card_id:
                self.previous_card_positions[o.card_id] = o.center
        return result