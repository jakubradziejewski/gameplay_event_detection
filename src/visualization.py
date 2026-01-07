import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# Color constants
TEAM_COLORS = {'A': (255, 0, 0), 'B': (0, 255, 0)}
TOKEN_COLORS = {'A': (255, 0, 255), 'B': (255, 255, 0)}
BATTLE_COLOR = (0, 0, 255)
BOARD_COLOR = (255, 255, 0)


def draw_dice(frame, dice_list):
    """
    Draws markers on each die and a single info block.
    Shows individual scores only, no totals or sums.
    """
    # 1. Draw individual markers on the dice in the image
    for die in dice_list:
        x, y, value = die['x'], die['y'], die['value']
        
        # Circle on the die
        cv2.circle(frame, (x, y), 18, (0, 255, 0), 2)
        
        # Value label directly above each die
        cv2.putText(
            frame, 
            str(value), 
            (x - 10, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 0, 255), 
            2
        )

    # Starting Y position
    start_y = 120 
    
    # Semi-transparent background for the text block
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, start_y), (350, start_y + 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Dice Count
    cv2.putText(frame, f'Total Dice: {len(dice_list)}', (20, start_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Individual Scores List
    if dice_list:
        scores = [str(die['value']) for die in dice_list]
        score_str = f"Scores: {', '.join(scores)}"
        cv2.putText(frame, score_str, (20, start_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        
def draw_text_with_bg(frame: np.ndarray, text: str, pos: Tuple[int, int], 
                     color: Tuple[int, int, int], font_scale: float = 0.7, 
                     thickness: int = 2, bg_alpha: float = 0.7):
    """Draw text with a semi-transparent background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    
    # Draw background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (x-5, y-th-5), (x+tw+5, y+5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1-bg_alpha, 0, frame)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def get_top_point(box: np.ndarray) -> np.ndarray:
    """Get the topmost point of a bounding box"""
    return box[np.argmin(box[:, 1])]

def get_bottom_point(box: np.ndarray) -> np.ndarray:
    """Get the bottom-most point of a bounding box"""
    return box[np.argmax(box[:, 1])]

def draw_scoreboard(frame: np.ndarray, scores: Dict[str, int], width: int):
    """Draw the scoreboard at the top of the frame"""

    # Semi-transparent black background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Team A score (left)
    cv2.putText(frame, f"Team A: {scores['A']}", (50, 40), 
               font, 1.2, TEAM_COLORS['A'], 3)
    
    # Team B score (right)
    team_b_text = f"Team B: {scores['B']}"
    (tw, _), _ = cv2.getTextSize(team_b_text, font, 1.2, 3)
    cv2.putText(frame, team_b_text, (width-tw-50, 40), 
               font, 1.2, TEAM_COLORS['B'], 3)
    
    # Center divider line
    cv2.line(frame, (width//2, 0), (width//2, 60), (255, 255, 255), 2)


def draw_events(frame: np.ndarray, events: List[str], width: int):
    """Draw the last two event messages"""
    if not events:
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 80
    
    # Get only the last 2 events
    events_to_show = events[-2:] if len(events) >= 2 else events
    
    # Draw them stacked vertically
    for event in events_to_show:
        (tw, th), _ = cv2.getTextSize(event, font, 0.8, 2)
        x = (width - tw) // 2
        
        draw_text_with_bg(frame, event, (x, y), (255, 255, 255), 
                          font_scale=0.8, bg_alpha=0.8)
        
        y += 35  # Move down for next message


def draw_token_messages(frame: np.ndarray, messages: Dict[int, str], 
                       width: int, height: int):
    """Draw token hit messages"""
    if not messages:
        return
    
    # Start at 90% down the screen
    y_offset = int(height * 0.9)
    
    for battle_id, message in messages.items():
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(message, font, 0.6, 2)
        x = (width - tw) // 2
        
        draw_text_with_bg(frame, message, (x, y_offset), (255, 100, 255), 
                          font_scale=0.6, bg_alpha=0.8)
        y_offset += 30


def draw_board(frame: np.ndarray, board_corners: np.ndarray):
    """Draw the board boundary"""
    if board_corners is None:
        return
    
    cv2.polylines(frame, [board_corners.astype(np.int32)], True, BOARD_COLOR, 2)
    cv2.putText(frame, "Board", 
               (int(board_corners[0][0]+10), int(board_corners[0][1]+30)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOARD_COLOR, 2)


def draw_cards(frame: np.ndarray, cards: List):
    """Draw all detected cards with labels"""
    for card in cards:
        color = TEAM_COLORS.get(card.team, (0, 255, 0))
        
        # Draw card box
        cv2.drawContours(frame, [card.box], 0, color, 3)
        
        # Get top-left corner of the bounding box
        top_left = card.box[np.argmin(card.box[:, 0] + card.box[:, 1])]
        
        # Create label
        label = f"Card {card.card_id}"
        if card.team:
            label += f" ({card.team})"
        
        # Draw label at top-left corner
        draw_text_with_bg(frame, label, (top_left[0] + 5, top_left[1] + 20), 
                          color, font_scale=0.6)

def draw_battles(frame: np.ndarray, battles: List, token_detector=None):
    """Draw all active battles with labels and optional tokens"""
    for battle in battles:
        # Draw battle box
        cv2.drawContours(frame, [battle.box], 0, BATTLE_COLOR, 3)
        
        # Draw tokens if token_detector is provided
        if token_detector:
            token_detector.draw_tokens(frame, battle.battle_id)
        
        # Draw battle label
        bottom = get_bottom_point(battle.box)
        text = f"Battle: {battle.card1_id}({battle.team1}) vs {battle.card2_id}({battle.team2})"
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_pos = (bottom[0] - tw // 2, bottom[1] + th + 10)
        
        draw_text_with_bg(frame, text, text_pos, BATTLE_COLOR)

def draw_stats(frame: np.ndarray, frame_count: int, total_frames: int, 
               num_cards: int, num_battles: int, height: int):
    """Draw statistics at the bottom of the frame"""
    stats = f"Frame: {frame_count}/{total_frames} | Cards: {num_cards} | Battles: {num_battles}"
    cv2.putText(frame, stats, (10, height-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


