import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

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

class CardDetector:
    def __init__(self, params=None):
        self.params = params or DetectionParams()
        self.next_id = 1
        self.tracked_cards = {}  # {card_id: (center, frame_count)}
        self.id_timeout = 30  # frames before ID is released

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
        
        # Remove single cards that overlap with battles
        cards = self._remove_overlapping_cards(cards, battles)
        
        # Assign IDs only to single cards
        self._assign_ids(cards)
        
        return cards, battles

    def _remove_overlapping_cards(self, cards: List[Card], battles: List[Card]) -> List[Card]:
        """Remove single cards that overlap with battle areas"""
        non_overlapping = []
        
        for card in cards:
            overlaps = False
            for battle in battles:
                # Check if card center is inside battle box
                if cv2.pointPolygonTest(battle.box, card.center, False) >= 0:
                    overlaps = True
                    break
                
                # Check if battle center is inside card box
                if cv2.pointPolygonTest(card.box, battle.center, False) >= 0:
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(card)
        
        return non_overlapping

    def _assign_ids(self, cards: List[Card]):
        """Assign persistent IDs to detected cards"""
        # Clean up old tracked cards
        self.tracked_cards = {
            cid: (center, count + 1) 
            for cid, (center, count) in self.tracked_cards.items() 
            if count < self.id_timeout
        }
        
        # Match current cards to tracked cards
        used_ids = set()
        for card in cards:
            best_match_id = None
            best_distance = float('inf')
            
            for card_id, (tracked_center, _) in self.tracked_cards.items():
                if card_id in used_ids:
                    continue
                    
                distance = np.sqrt(
                    (card.center[0] - tracked_center[0])**2 + 
                    (card.center[1] - tracked_center[1])**2
                )
                
                if distance < 100 and distance < best_distance:  # 100px threshold
                    best_distance = distance
                    best_match_id = card_id
            
            if best_match_id is not None:
                card.card_id = best_match_id
                used_ids.add(best_match_id)
                self.tracked_cards[best_match_id] = (card.center, 0)
            else:
                # Assign new ID
                card.card_id = self.next_id
                self.next_id += 1
                self.tracked_cards[card.card_id] = (card.center, 0)
                used_ids.add(card.card_id)