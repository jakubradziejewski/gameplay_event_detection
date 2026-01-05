import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

@dataclass
class DetectionParams:
    gradient_threshold: int = 30
    canny_low: int = 50
    canny_high: int = 150
    min_area: int = 9000
    max_area: int = 16000
    min_aspect: float = 1.2
    max_aspect: float = 2.8
    solidity_threshold: float = 0.85
    edge_buffer_x: float = 0.10
    edge_buffer_y: float = 0.18
    edge_min_area: int = 10000
    edge_solidity_threshold: float = 0.90
    battle_proximity_buffer: float = 0.10  # 10% buffer for battle detection

@dataclass
class Card:
    box: np.ndarray
    center: Tuple[float, float]
    width: float
    height: float
    card_id: Optional[int] = None
    team: Optional[str] = None
    is_ghost: bool = False
    in_edge_zone: bool = False

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
    def __init__(self, params=None):
        self.params = params or DetectionParams()
        self.next_id = 1
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.previous_card_positions: Dict[int, Tuple[float, float]] = {}
        self.battles: Dict[int, Battle] = {}
        self.next_battle_id = 1
        self.current_frame = 0
        self.board_center_x = None
        self.edge_buffer = None
        
    def _create_tracker(self):
        return cv2.legacy.TrackerKCF_create()

    def _assign_team(self, center: Tuple[float, float]) -> str:
        return 'A' if center[0] < self.board_center_x else 'B' if self.board_center_x else None
    
    def _calculate_edge_buffer(self, board_corners: np.ndarray):
        if board_corners is None:
            return None
        x_min, y_min = np.min(board_corners, axis=0)
        x_max, y_max = np.max(board_corners, axis=0)
        width, height = x_max - x_min, y_max - y_min
        buffer_x, buffer_y = width * self.params.edge_buffer_x, height * self.params.edge_buffer_y
        return {
            'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max,
            'inner_x_min': x_min + buffer_x, 'inner_y_min': y_min + buffer_y,
            'inner_x_max': x_max - buffer_x, 'inner_y_max': y_max - buffer_y
        }
    
    def _is_in_edge_zone(self, center: Tuple[float, float]) -> bool:
        if not self.edge_buffer:
            return False
        x, y = center
        eb = self.edge_buffer
        in_board = eb['x_min'] <= x <= eb['x_max'] and eb['y_min'] <= y <= eb['y_max']
        in_safe = eb['inner_x_min'] <= x <= eb['inner_x_max'] and eb['inner_y_min'] <= y <= eb['inner_y_max']
        return in_board and not in_safe

    def detect_cards(self, frame, board_corners=None):
        self.current_frame += 1
        if board_corners is not None:
            if self.board_center_x is None:
                self.board_center_x = np.mean(board_corners[:, 0])
            if self.edge_buffer is None:
                self.edge_buffer = self._calculate_edge_buffer(board_corners)
        
        self._update_trackers(frame)
        detected_cards = self._detect_cards_only(frame)
        
        if board_corners is not None:
            detected_cards = self._filter_inside_board(detected_cards, board_corners)
        
        detected_cards = self._filter_edge_cards(detected_cards)
        detected_cards = self._merge_overlaps(detected_cards)
        
        all_cards = self._match_and_track(frame, detected_cards, board_corners)
        all_cards = self._add_ghosts(all_cards)
        
        # Detect battles from card positions
        self._detect_battles_from_cards(all_cards)
        
        # Process battles (check if cards left battle zones)
        battle_results = self._process_battles(all_cards)
        
        return all_cards, battle_results
    
    def _detect_cards_only(self, frame):
        """Detect only individual cards, no battle shapes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        _, gbin = cv2.threshold(grad, self.params.gradient_threshold, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gbin, self.params.canny_low, self.params.canny_high)
        kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dil = cv2.dilate(edges, kern2, iterations=2)
        cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray)
        for c in cnts:
            if cv2.contourArea(c) > 1000:
                cv2.drawContours(mask, [c], -1, 255, -1)
        
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
        fg = np.uint8(fg)
        bg = cv2.dilate(mask, kern2, iterations=3)
        unk = cv2.subtract(bg, fg)
        _, mrk = cv2.connectedComponents(fg)
        mrk = mrk + 1
        mrk[unk == 255] = 0
        mrk = cv2.watershed(frame, mrk)
        
        cards = []
        for lbl in np.unique(mrk):
            if lbl <= 1:
                continue
            tmask = np.zeros_like(gray)
            tmask[mrk == lbl] = 255
            cs, _ = cv2.findContours(tmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            c = cs[0]
            area = cv2.contourArea(c)
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            if h < w:
                w, h = h, w
            asp = h / w
            hull = cv2.convexHull(c)
            sol = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            ctr = rect[0]
            edge = self._is_in_edge_zone(ctr)
            
            if edge and (sol < self.params.edge_solidity_threshold or area < self.params.edge_min_area):
                continue
            if not edge and sol < self.params.solidity_threshold:
                continue
            
            box = np.int32(cv2.boxPoints(rect))
            p = self.params
            # Only detect regular cards
            if p.min_area <= area <= p.max_area and p.min_aspect <= asp <= p.max_aspect:
                cards.append(Card(box=box, center=ctr, width=w, height=h, in_edge_zone=edge))
        
        return cards
    
    def _detect_battles_from_cards(self, cards: List[Card]):
        """Detect battles by checking if opposing team cards are close to each other"""
        # Only consider non-ghost cards
        active_cards = [c for c in cards if not c.is_ghost and c.team]
        
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
                            initial_center=battle_center
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
        """Check if battles have ended (cards moved too far apart)"""
        battles_to_remove = []
        battle_events = []
        
        for battle_id, battle in list(self.battles.items()):
            # Find the two cards involved
            card1 = next((c for c in cards if c.card_id == battle.card1_id), None)
            card2 = next((c for c in cards if c.card_id == battle.card2_id), None)
            
            # Check if both cards still exist
            card1_exists = card1 is not None and not card1.is_ghost
            card2_exists = card2 is not None and not card2.is_ghost
            
            if card1_exists and card2_exists:
                # Update battle bbox to follow cards as they move
                battle.bbox = self._create_battle_bbox(card1, card2)
                battle.box = self._bbox_to_box(battle.bbox)
                
                # Check if cards are still close enough to be in battle
                if not self._are_cards_in_battle(card1, card2):
                    # Cards moved too far apart - determine who lost
                    # The card that moved further from initial battle center loses
                    dist1 = np.sqrt((card1.center[0] - battle.initial_center[0])**2 + 
                                  (card1.center[1] - battle.initial_center[1])**2)
                    dist2 = np.sqrt((card2.center[0] - battle.initial_center[0])**2 + 
                                  (card2.center[1] - battle.initial_center[1])**2)
                    
                    if dist1 > dist2:
                        # Card 1 moved away and lost
                        battles_to_remove.append(battle_id)
                        battle_events.append(f"Card {battle.card1_id} ({battle.team1}) lost the battle")
                    else:
                        # Card 2 moved away and lost
                        battles_to_remove.append(battle_id)
                        battle_events.append(f"Card {battle.card2_id} ({battle.team2}) lost the battle")
            elif card1_exists and not card2_exists:
                # Card 2 disappeared - Card 2 lost
                battles_to_remove.append(battle_id)
                battle_events.append(f"Card {battle.card2_id} ({battle.team2}) lost the battle")
            elif card2_exists and not card1_exists:
                # Card 1 disappeared - Card 1 lost
                battles_to_remove.append(battle_id)
                battle_events.append(f"Card {battle.card1_id} ({battle.team1}) lost the battle")
            else:
                # Both cards disappeared - just end battle
                battles_to_remove.append(battle_id)
        
        # Remove ended battles
        for battle_id in battles_to_remove:
            del self.battles[battle_id]
        
        # Store events for display
        self.battle_events = battle_events
        
        # Return active battles as list
        return list(self.battles.values())
    
    def _is_card_in_battle_zone(self, card: Card, battle: Battle) -> bool:
        """Check if card center is within battle bounding box"""
        bx, by, bw, bh = battle.bbox
        cx, cy = card.center
        
        return bx <= cx <= bx + bw and by <= cy <= by + bh
    
    def _filter_edge_cards(self, detections):
        filtered = []
        for det in detections:
            det.in_edge_zone = self._is_in_edge_zone(det.center)
            if not det.in_edge_zone:
                filtered.append(det)
            else:
                bbox = self._box_to_bbox(det.box)
                if any(self._iou(bbox, o.bbox) > 0.3 for o in self.tracked_objects.values()):
                    filtered.append(det)
        return filtered
    
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
    
    def _add_ghosts(self, objs):
        visible = {o.card_id for o in objs}
        for oid, t in self.tracked_objects.items():
            if oid not in visible and 0 < t.frames_lost <= 20:
                if t.last_valid_bbox and t.last_valid_box is not None:
                    x, y, w, h = t.last_valid_bbox
                    objs.append(Card(box=t.last_valid_box.copy(), center=(x+w/2,y+h/2),
                                   width=w, height=h, card_id=oid, team=t.team, is_ghost=True))
        return objs
    
    def get_team_scores(self, cards, battles):
        scores = {'A': 0, 'B': 0}
        for c in cards:
            if c.team and not c.is_ghost:
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
                if t.frames_lost > 30:
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
                if det.in_edge_zone:
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