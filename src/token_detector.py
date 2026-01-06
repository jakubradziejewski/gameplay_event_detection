import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

@dataclass
class Token:
    center: Tuple[float, float]
    radius: float
    color_value: Tuple[int, int, int]  # BGR color
    team: str  # 'A' or 'B'
    battle_id: int  # Which battle this token belongs to
    token_id: int  # Unique ID for this token

@dataclass
class TrackedToken:
    token_id: int
    tracker: any
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    radius: float
    team: str
    battle_id: int
    frames_tracked: int = 0  # How many frames this token has been tracked
    frames_lost: int = 0
    last_center: Tuple[float, float] = None
    is_confirmed: bool = False  # Whether this token is confirmed (stable for N frames)

class TokenDetector:
    def __init__(self):
        # HSV range for red color (red wraps around in HSV)
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])
        
        # Size parameters (relative to board size)
        self.min_radius_ratio = 0.011
        self.max_radius_ratio = 0.015
        
        # Hough Circle parameters
        self.hough_dp = 1
        self.hough_min_dist_ratio = 0.02
        self.hough_param1 = 50
        self.hough_param2 = 9
        self.hough_edge_blur = 3
        
        # Battle token tracking
        self.battle_confirmed_tokens: Dict[int, List[Token]] = {}  # battle_id -> list of CONFIRMED tokens
        self.battle_tokens = self.battle_confirmed_tokens  # Alias for backward compatibility
        self.battle_hit_cards: Dict[int, Dict[str, int]] = {}  # battle_id -> {card_id: token_count}
        
        # Token tracking with immediate tracking
        self.tracked_tokens: Dict[int, TrackedToken] = {}  # token_id -> TrackedToken (all tokens, confirmed or not)
        self.next_token_id = 1
        self.confirmation_threshold = 15  # Frames required for confirmation
        self.max_token_lost_frames = 10  # Remove token after this many lost frames
        self.max_match_distance = 20.0  # Maximum distance to match detection to tracked token
        self.current_frame = 0
    
    def _create_tracker(self):
        """Create a new KCF tracker for token tracking"""
        return cv2.legacy.TrackerKCF_create()
    
    def _calculate_board_size(self, board_corners: np.ndarray) -> float:
        """Calculate board width for size reference"""
        if board_corners is None:
            return None
        
        left_center = (board_corners[0] + board_corners[3]) / 2
        right_center = (board_corners[1] + board_corners[2]) / 2
        width = np.linalg.norm(right_center - left_center)
        return width
    
    def _get_battle_search_region(self, battle_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Calculate the region above and at edges of battle where tokens should be detected
        
        Extends search to include tokens on the edge and slightly inside the battle boundary
        
        Returns: (x, y, w, h) of search region
        """
        bx, by, bw, bh = battle_bbox
        
        # Extended region: 
        # - Extends 10% above the battle top edge (to catch tokens on top edge)
        # - Extends 50% of battle height above that
        # - Includes 10% inside the battle boundaries on sides and top
        margin_above = int(bh * 0.05)  # Small margin above battle
        search_height = int(bh * 0.5)  # Total search height (includes margin + extension above)
        
        # Extend search region slightly into the battle area (10% on each side and top)
        edge_extension = int(bw * 0.05)
        top_extension = int(bh * 0.05)
        
        # Start search region above the battle, but extend into it slightly
        search_y = max(0, by - margin_above - search_height + top_extension)
        search_x = max(0, bx - edge_extension)
        search_w = bw + (2 * edge_extension)
        search_h = search_height + top_extension  # Includes the top portion of battle
        
        return (search_x, search_y, search_w, search_h)
    
    def _create_region_mask(self, frame_shape: Tuple[int, int], 
                           region: Tuple[int, int, int, int]) -> np.ndarray:
        """Create a mask for the search region"""
        mask = np.zeros(frame_shape, dtype=np.uint8)
        x, y, w, h = region
        
        # Ensure bounds are valid
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_shape[1] - x)
        h = min(h, frame_shape[0] - y)
        
        if w > 0 and h > 0:
            mask[y:y+h, x:x+w] = 255
        
        return mask
    
    def _update_token_trackers(self, frame: np.ndarray):
        """Update all token trackers and remove lost tokens"""
        to_remove = []
        
        for token_id, tracked_token in list(self.tracked_tokens.items()):
            ok, bbox = tracked_token.tracker.update(frame)
            
            if ok:
                tracked_token.bbox = tuple(map(int, bbox))
                tracked_token.frames_lost = 0
                tracked_token.frames_tracked += 1
                x, y, w, h = tracked_token.bbox
                tracked_token.center = (x + w/2, y + h/2)
                tracked_token.last_center = tracked_token.center
                
                # Check if token should be confirmed now
                if not tracked_token.is_confirmed and tracked_token.frames_tracked >= self.confirmation_threshold:
                    tracked_token.is_confirmed = True
                    self._confirm_token(tracked_token)
            else:
                tracked_token.frames_lost += 1
                
                # Remove token if lost for too long
                if tracked_token.frames_lost > self.max_token_lost_frames:
                    to_remove.append(token_id)
        
        for token_id in to_remove:
            # Remove from confirmed tokens if it was confirmed
            tracked_token = self.tracked_tokens[token_id]
            if tracked_token.is_confirmed and tracked_token.battle_id in self.battle_confirmed_tokens:
                self.battle_confirmed_tokens[tracked_token.battle_id] = [
                    t for t in self.battle_confirmed_tokens[tracked_token.battle_id] 
                    if t.token_id != token_id
                ]
            
            del self.tracked_tokens[token_id]
    
    def _confirm_token(self, tracked_token: TrackedToken):
        """Confirm a token and add it to the confirmed list, update hit counts"""
        battle_id = tracked_token.battle_id
        
        # Initialize confirmed tokens list if needed
        if battle_id not in self.battle_confirmed_tokens:
            self.battle_confirmed_tokens[battle_id] = []
        
        # Create confirmed token
        token = Token(
            center=tracked_token.center,
            radius=tracked_token.radius,
            color_value=(0, 0, 255),  # Red default
            team=tracked_token.team,
            battle_id=battle_id,
            token_id=tracked_token.token_id
        )
        
        self.battle_confirmed_tokens[battle_id].append(token)
        
        # Update hit count - this is handled separately via update_hit_counts()
    
    def _match_detection_to_tracked_tokens(self, detection: Dict, battle_id: int) -> Optional[int]:
        """Try to match a detection to an existing tracked token
        
        Returns: token_id if matched, None otherwise
        """
        best_match_id = None
        best_distance = float('inf')
        
        for token_id, tracked_token in self.tracked_tokens.items():
            # Only match tokens from the same battle and team
            if tracked_token.battle_id != battle_id or tracked_token.team != detection['team']:
                continue
            
            # Calculate distance
            dist = np.sqrt((tracked_token.center[0] - detection['center'][0])**2 + 
                          (tracked_token.center[1] - detection['center'][1])**2)
            
            if dist < best_distance and dist <= self.max_match_distance:
                best_distance = dist
                best_match_id = token_id
        
        return best_match_id
    
    def detect_tokens_for_battles(self, frame: np.ndarray, battles: List, 
                                  board_corners: Optional[np.ndarray] = None) -> Dict[int, List[Dict]]:
        """Detect tokens in regions above active battles
        
        Returns: Dictionary mapping battle_id to list of detected token candidates
        """
        self.current_frame += 1
        
        # Update existing token trackers first
        self._update_token_trackers(frame)
        
        if not battles:
            return {}
        
        # Calculate board size for scaling
        board_width = self._calculate_board_size(board_corners)
        if board_width is None:
            board_width = frame.shape[1]
        
        min_radius = int(board_width * self.min_radius_ratio)
        max_radius = int(board_width * self.max_radius_ratio)
        min_dist = int(board_width * self.hough_min_dist_ratio)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create red mask
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations with circular kernels
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        red_mask = cv2.dilate(red_mask, kernel_small, iterations=1)
        
        # Apply Gaussian blur
        if self.hough_edge_blur > 0:
            red_mask = cv2.GaussianBlur(red_mask, (self.hough_edge_blur, self.hough_edge_blur), 0)
        
        battle_token_candidates = {}
        
        for battle in battles:
            # Use new battle structure
            battle_id = battle.battle_id
            bx, by, bw, bh = battle.bbox
            
            # Get search region
            search_region = self._get_battle_search_region((bx, by, bw, bh))
            sx, sy, sw, sh = search_region
            
            # Create region mask
            region_mask = self._create_region_mask(frame.shape[:2], search_region)
            
            # Apply region mask to red mask
            red_mask_region = cv2.bitwise_and(red_mask, region_mask)
            
            # Detect circles in this region
            circles = cv2.HoughCircles(
                red_mask_region,
                cv2.HOUGH_GRADIENT,
                dp=self.hough_dp,
                minDist=min_dist,
                param1=self.hough_param1,
                param2=self.hough_param2,
                minRadius=min_radius,
                maxRadius=max_radius
            )
            
            candidates = []
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                
                # Calculate battle center x for team assignment
                battle_center_x = bx + bw / 2
                
                for circle in circles[0, :]:
                    cx, cy, radius = circle
                    center = (float(cx), float(cy))
                    
                    # Determine team based on which side of battle center the token is
                    if cx < battle_center_x:
                        team = 'A'
                    else:
                        team = 'B'
                    
                    # Get average color
                    mask_single = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.circle(mask_single, (cx, cy), radius, 255, -1)
                    mean_color = cv2.mean(frame, mask=mask_single)[:3]
                    
                    candidates.append({
                        'center': center,
                        'radius': float(radius),
                        'color_value': mean_color,
                        'team': team,
                        'battle_id': battle_id
                    })
            
            if candidates:
                battle_token_candidates[battle_id] = candidates
        
        return battle_token_candidates
    
    def update_battle_tokens(self, battle_token_candidates: Dict[int, List[Dict]], 
                            battles: List, tracked_objects: Dict, frame: np.ndarray) -> None:
        """Update token tracking - track immediately, confirm after stability threshold"""
        
        # Get set of active battles
        active_battles = {b.battle_id for b in battles}
        
        # Remove tokens for finished battles
        finished_battles = set(self.battle_confirmed_tokens.keys()) - active_battles
        for battle_id in finished_battles:
            if battle_id in self.battle_confirmed_tokens:
                del self.battle_confirmed_tokens[battle_id]
            if battle_id in self.battle_hit_cards:
                del self.battle_hit_cards[battle_id]
            
            # Remove tracked tokens for this battle
            tokens_to_remove = [tid for tid, tt in self.tracked_tokens.items() if tt.battle_id == battle_id]
            for tid in tokens_to_remove:
                del self.tracked_tokens[tid]
        
        # Process each battle's candidate tokens
        for battle_id in active_battles:
            # Initialize tracking structures for new battle
            if battle_id not in self.battle_confirmed_tokens:
                self.battle_confirmed_tokens[battle_id] = []
            if battle_id not in self.battle_hit_cards:
                self.battle_hit_cards[battle_id] = {}
            
            # Get current frame's detections
            current_detections = battle_token_candidates.get(battle_id, [])
            
            # Try to match each detection to existing tracked tokens
            for detection in current_detections:
                matched_token_id = self._match_detection_to_tracked_tokens(detection, battle_id)
                
                # If no match found, start tracking this as a new token immediately
                if matched_token_id is None:
                    token_id = self.next_token_id
                    self.next_token_id += 1
                    
                    # Create bounding box from center and radius
                    cx, cy = detection['center']
                    radius = detection['radius']
                    bbox = (int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))
                    
                    # Initialize tracker immediately
                    tracker = self._create_tracker()
                    tracker.init(frame, bbox)
                    
                    # Create tracked token (not yet confirmed)
                    tracked_token = TrackedToken(
                        token_id=token_id,
                        tracker=tracker,
                        bbox=bbox,
                        center=detection['center'],
                        radius=detection['radius'],
                        team=detection['team'],
                        battle_id=battle_id,
                        last_center=detection['center'],
                        frames_tracked=1,
                        is_confirmed=False
                    )
                    self.tracked_tokens[token_id] = tracked_token
        
        # Update hit counts for confirmed tokens
        self._update_hit_counts(battles, tracked_objects)
    
    def _update_hit_counts(self, battles: List, tracked_objects: Dict):
        """Update hit counts based on confirmed tokens"""
        for battle in battles:
            if battle.battle_id not in self.battle_confirmed_tokens:
                continue
            
            battle_id = battle.battle_id
            c1_id = battle.card1_id
            c2_id = battle.card2_id
            
            # Count confirmed tokens by team
            confirmed_tokens = self.battle_confirmed_tokens[battle_id]
            team_counts = {'A': 0, 'B': 0}
            
            for token in confirmed_tokens:
                if token.team in team_counts:
                    team_counts[token.team] += 1
            
            # Get card teams from tracked objects
            c1_team = tracked_objects.get(c1_id)
            c2_team = tracked_objects.get(c2_id)
            
            # Update hit counts based on team
            if battle_id not in self.battle_hit_cards:
                self.battle_hit_cards[battle_id] = {}
            
            for team, count in team_counts.items():
                if count > 0:
                    # Determine which card was hit
                    if c1_team and hasattr(c1_team, 'team') and c1_team.team == team:
                        hit_card_id = c1_id
                    elif c2_team and hasattr(c2_team, 'team') and c2_team.team == team:
                        hit_card_id = c2_id
                    else:
                        hit_card_id = c1_id if team == 'A' else c2_id
                    
                    self.battle_hit_cards[battle_id][hit_card_id] = count
    
    def get_battle_messages(self, battles: List) -> Dict[int, str]:
        """Get hit messages for each battle"""
        messages = {}
        
        for battle in battles:
            if battle.battle_id not in self.battle_hit_cards:
                continue
            
            battle_id = battle.battle_id
            hit_counts = self.battle_hit_cards[battle_id]
            
            if not hit_counts:
                continue
            
            # Create message for each hit card
            message_parts = []
            for card_id, count in hit_counts.items():
                # Get card team from battle participants
                c1_id = battle.card1_id
                c2_id = battle.card2_id
                
                if card_id == c1_id:
                    team = battle.team1
                    message_parts.append(f"Card {card_id} ({team}) got hit by {count} token{'s' if count != 1 else ''}")
                elif card_id == c2_id:
                    team = battle.team2
                    message_parts.append(f"Card {card_id} ({team}) got hit by {count} token{'s' if count != 1 else ''}")
            
            if message_parts:
                messages[battle_id] = " | ".join(message_parts)
        
        return messages
    
    def draw_tokens(self, frame: np.ndarray, battle_id: int, show_unconfirmed: bool = False) -> None:
        """Draw confirmed tokens and optionally unconfirmed tokens for a specific battle"""
        TEAM_COLORS = {'A': (255, 0, 255), 'B': (255, 255, 0)}  # Violet for A, Cyan for B
        
        # Draw all tracked tokens for this battle
        for token_id, tracked_token in self.tracked_tokens.items():
            if tracked_token.battle_id != battle_id:
                continue
            
            cx, cy = int(tracked_token.center[0]), int(tracked_token.center[1])
            radius = int(tracked_token.radius)
            color = TEAM_COLORS.get(tracked_token.team, (0, 255, 0))
            
            if tracked_token.is_confirmed:
                # Draw confirmed tokens with solid circle
                cv2.circle(frame, (cx, cy), radius, color, 2)
                cv2.circle(frame, (cx, cy), 2, color, -1)
            elif show_unconfirmed:
                # Draw unconfirmed tokens with dimmer, dashed style
                dim_color = tuple(int(c * 0.5) for c in color)
                cv2.circle(frame, (cx, cy), radius, dim_color, 1)
                
                # Show stability progress
                stability_text = f"{tracked_token.frames_tracked}/{self.confirmation_threshold}"
                cv2.putText(frame, stability_text, (cx + radius + 2, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, dim_color, 1)
    
    def draw_search_regions(self, frame: np.ndarray, battles: List) -> None:
        """Draw search regions for debugging"""
        for battle in battles:
            bx, by, bw, bh = battle.bbox
            
            search_region = self._get_battle_search_region((bx, by, bw, bh))
            sx, sy, sw, sh = search_region
            
            # Draw search region
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 0, 255), 1)