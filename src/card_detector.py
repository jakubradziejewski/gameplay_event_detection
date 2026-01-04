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
    battle_min_area: int = 18000
    battle_max_area: int = 34000
    battle_min_aspect: float = 0.4
    battle_max_aspect: float = 1.3
    battle_confirmation_frames: int = 5

@dataclass
class Card:
    box: np.ndarray
    center: Tuple[float, float]
    width: float
    height: float
    card_id: Optional[int] = None
    is_battle: bool = False
    battle_ids: Optional[Tuple[int, int]] = None
    team: Optional[str] = None
    is_ghost: bool = False
    in_edge_zone: bool = False

@dataclass
class TrackedObject:
    object_id: int
    tracker: any
    bbox: Tuple[int, int, int, int]
    is_battle: bool
    battle_ids: Optional[Tuple[int, int]] = None
    frames_lost: int = 0
    last_center: Tuple[float, float] = None
    team: Optional[str] = None
    initial_bbox: Optional[Tuple[int, int, int, int]] = None
    initial_card_positions: Optional[Dict[int, Tuple[float, float]]] = None
    last_valid_bbox: Optional[Tuple[int, int, int, int]] = None
    last_valid_box: Optional[np.ndarray] = None
    is_confirmed_battle: bool = False
    battle_confirmation_count: int = 0
    was_in_battle: bool = False
    battle_id_involved: Optional[int] = None
    frames_both_missing: int = 0
    frames_one_missing: int = 0
    missing_card_id: Optional[int] = None

class CardDetector:
    def __init__(self, params=None):
        self.params = params or DetectionParams()
        self.next_id = 1
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.previous_card_positions: Dict[int, Tuple[float, float]] = {}
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
        detected_cards, detected_battles = self._detect_objects(frame)
        
        if board_corners is not None:
            detected_cards = self._filter_inside_board(detected_cards, board_corners)
            detected_battles = self._filter_inside_board(detected_battles, board_corners)
        
        detected_cards = self._filter_edge_cards(detected_cards)
        detected_cards = self._filter_battle_zones(detected_cards)
        detected_battles = self._filter_battle_zones(detected_battles)
        detected_cards = self._merge_overlaps(detected_cards)
        
        all_objects = self._match_and_track(frame, detected_cards, detected_battles, board_corners)
        all_objects = self._process_battles(all_objects)
        all_objects = self._add_ghosts(all_objects)
        
        return [o for o in all_objects if not o.is_battle], [o for o in all_objects if o.is_battle]
    
    def _filter_edge_cards(self, detections):
        filtered = []
        for det in detections:
            det.in_edge_zone = self._is_in_edge_zone(det.center)
            if not det.in_edge_zone:
                filtered.append(det)
            else:
                bbox = self._box_to_bbox(det.box)
                if any(not o.is_battle and self._iou(bbox, o.bbox) > 0.3 for o in self.tracked_objects.values()):
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
            if oid not in visible and not t.is_battle and 0 < t.frames_lost <= 20:
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
    
    def _filter_battle_zones(self, objs):
        battles = [o for o in self.tracked_objects.values() if o.is_battle and o.is_confirmed_battle]
        if not battles:
            return objs
        filtered = []
        for obj in objs:
            overlaps = False
            for b in battles:
                bx, by, bw, bh = b.bbox
                dist = np.sqrt((obj.center[0]-(bx+bw/2))**2 + (obj.center[1]-(by+bh/2))**2)
                if dist < np.sqrt(bw**2 + bh**2) * 0.6:
                    overlaps = True
                    break
            if not overlaps:
                filtered.append(obj)
        return filtered
    
    def _process_battles(self, objs):
        to_end, to_remove, to_restore = [], [], []
        for battle in [o for o in objs if o.is_battle]:
            if not battle.battle_ids or battle.card_id not in self.tracked_objects:
                continue
            tb = self.tracked_objects[battle.card_id]
            c1, c2 = battle.battle_ids
            
            for cid in [c1, c2]:
                if cid in self.tracked_objects:
                    self.tracked_objects[cid].was_in_battle = True
                    self.tracked_objects[cid].battle_id_involved = battle.card_id
            
            visible = {o.card_id for o in objs if not o.is_battle}
            c1_vis, c2_vis = c1 in visible, c2 in visible
            c1_ex, c2_ex = c1 in self.tracked_objects, c2 in self.tracked_objects
            
            if not tb.is_confirmed_battle:
                if c1_vis and c2_vis:
                    tb.battle_confirmation_count += 1
                    if tb.battle_confirmation_count >= self.params.battle_confirmation_frames:
                        tb.is_confirmed_battle = True
                else:
                    tb.battle_confirmation_count = 0
                    if tb.frames_lost > 3:
                        to_remove.append(battle.card_id)
                continue
            
            c1_pos = c2_pos = False
            if c1_ex and (t1 := self.tracked_objects[c1]).last_center:
                c1_pos = np.sqrt((t1.last_center[0]-tb.last_center[0])**2+(t1.last_center[1]-tb.last_center[1])**2) < 100
            if c2_ex and (t2 := self.tracked_objects[c2]).last_center:
                c2_pos = np.sqrt((t2.last_center[0]-tb.last_center[0])**2+(t2.last_center[1]-tb.last_center[1])**2) < 100
            
            if (not c1_ex or not c1_pos) and (not c2_ex or not c2_pos):
                to_remove.append(battle.card_id)
            elif not c1_ex or not c1_pos:
                to_end.append((battle.card_id, c2, c1))
            elif not c2_ex or not c2_pos:
                to_end.append((battle.card_id, c1, c2))
        
        for bid in to_remove:
            if (tb := self.tracked_objects.get(bid)) and tb.battle_ids:
                for cid in tb.battle_ids:
                    if cid in self.tracked_objects:
                        self.tracked_objects[cid].was_in_battle = False
                        self.tracked_objects[cid].battle_id_involved = None
            if bid in self.tracked_objects:
                del self.tracked_objects[bid]
            objs = [o for o in objs if o.card_id != bid]
        
        for bid, winner, loser in to_end:
            if (tb := self.tracked_objects.get(bid)) and tb.initial_card_positions:
                if (wpos := tb.initial_card_positions.get(winner)):
                    team = self.tracked_objects[winner].team if winner in self.tracked_objects else None
                    bx, by, bw, bh = tb.initial_bbox
                    bbox = (int(wpos[0]-bw/4), int(wpos[1]-bh/2), int(bw/2), int(bh))
                    tracker = self._create_tracker()
                    self.tracked_objects[winner] = TrackedObject(
                        object_id=winner, tracker=tracker, bbox=bbox, is_battle=False,
                        last_center=wpos, team=team
                    )
                    to_restore.append((winner, bbox, wpos, team))
            
            for d in [self.tracked_objects, self.previous_card_positions]:
                d.pop(loser, None)
            self.tracked_objects.pop(bid, None)
            objs = [o for o in objs if o.card_id != bid]
        
        for wid, bbox, pos, team in to_restore:
            x, y, w, h = bbox
            objs.append(Card(box=self._bbox_to_box(bbox), center=pos, width=w, height=h,
                           card_id=wid, team=team))
        return objs
    
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
                if not t.is_battle:
                    t.last_valid_bbox, t.last_valid_box = t.bbox, self._bbox_to_box(t.bbox)
            else:
                t.frames_lost += 1
                if not t.is_battle and t.frames_lost > (40 if t.is_battle else 30):
                    to_remove.append(oid)
        for oid in to_remove:
            self.tracked_objects.pop(oid, None)
            self.previous_card_positions.pop(oid, None)
    
    def _detect_objects(self, frame):
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
        
        cards, battles = [], []
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
            if p.battle_min_area <= area <= p.battle_max_area and p.battle_min_aspect <= asp <= p.battle_max_aspect:
                battles.append(Card(box=box, center=ctr, width=w, height=h, is_battle=True))
            elif p.min_area <= area <= p.max_area and p.min_aspect <= asp <= p.max_aspect:
                cards.append(Card(box=box, center=ctr, width=w, height=h, in_edge_zone=edge))
        return cards, battles

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
    
    def _match_and_track(self, frame, cards, battles, board):
        matched, result = set(), []
        for det in cards + battles:
            bbox = self._box_to_bbox(det.box)
            best_id, best_iou = None, 0.3
            for oid, t in self.tracked_objects.items():
                if oid in matched or t.is_battle != det.is_battle:
                    continue
                if (iou := self._iou(bbox, t.bbox)) > best_iou:
                    best_iou, best_id = iou, oid
            
            if best_id:
                t = self.tracked_objects[best_id]
                t.tracker = self._create_tracker()
                t.tracker.init(frame, bbox)
                t.bbox, t.frames_lost = bbox, 0
                if not t.is_battle:
                    t.last_valid_bbox, t.last_valid_box = bbox, det.box.copy()
                det.card_id, det.battle_ids, det.team = best_id, t.battle_ids, t.team
                result.append(det)
                matched.add(best_id)
            else:
                if not det.is_battle and any(not o.is_battle and self._overlap_ratio(bbox, o.bbox) > 0.1 
                                            for o in self.tracked_objects.values()):
                    continue
                if det.in_edge_zone and not det.is_battle:
                    continue
                if board is None or all(cv2.pointPolygonTest(board.astype(np.int32), (float(c[0]), float(c[1])), False) >= 0 
                                       for c in det.box):
                    nid = self.next_id
                    self.next_id += 1
                    tr = self._create_tracker()
                    tr.init(frame, bbox)
                    team = None if det.is_battle else self._assign_team(det.center)
                    bids, ipos = None, None
                    if det.is_battle:
                        bids = self._find_battle_cards(det.center)
                        if bids:
                            ipos = {c: self.previous_card_positions[c] for c in bids if c in self.previous_card_positions}
                    self.tracked_objects[nid] = TrackedObject(
                        object_id=nid, tracker=tr, bbox=bbox, is_battle=det.is_battle,
                        battle_ids=bids, last_center=det.center, team=team,
                        initial_bbox=bbox if det.is_battle else None,
                        initial_card_positions=ipos,
                        last_valid_bbox=bbox if not det.is_battle else None,
                        last_valid_box=det.box.copy() if not det.is_battle else None
                    )
                    det.card_id, det.battle_ids, det.team = nid, bids, team
                    result.append(det)
        
        for oid, t in list(self.tracked_objects.items()):
            if oid not in matched and t.frames_lost == 0:
                if board is None or all(cv2.pointPolygonTest(board.astype(np.int32), (float(x), float(y)), False) >= 0 
                                       for x, y in [(t.bbox[0], t.bbox[1]), (t.bbox[0]+t.bbox[2], t.bbox[1]),
                                                   (t.bbox[0]+t.bbox[2], t.bbox[1]+t.bbox[3]), (t.bbox[0], t.bbox[1]+t.bbox[3])]):
                    x, y, w, h = t.bbox
                    result.append(Card(box=self._bbox_to_box(t.bbox), center=(x+w/2, y+h/2),
                                     width=w, height=h, card_id=oid, is_battle=t.is_battle,
                                     battle_ids=t.battle_ids, team=t.team))
        
        for o in result:
            if not o.is_battle and o.card_id:
                self.previous_card_positions[o.card_id] = o.center
        return result
    
    def _find_battle_cards(self, center):
        nearby = [(oid, np.sqrt((t.last_center[0]-center[0])**2+(t.last_center[1]-center[1])**2))
                  for oid, t in self.tracked_objects.items() if not t.is_battle and t.last_center and 
                  np.sqrt((t.last_center[0]-center[0])**2+(t.last_center[1]-center[1])**2) < 150]
        nearby.sort(key=lambda x: x[1])
        return (nearby[0][0], nearby[1][0]) if len(nearby) >= 2 else None