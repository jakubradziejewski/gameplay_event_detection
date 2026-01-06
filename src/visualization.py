import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# Color constants
TEAM_COLORS = {'A': (255, 0, 0), 'B': (0, 255, 0)}
TOKEN_COLORS = {'A': (255, 0, 255), 'B': (255, 255, 0)}  # Violet for A, Cyan for B
BATTLE_COLOR = (0, 0, 255)
BOARD_COLOR = (255, 255, 0)


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
    """Draw the most recent event message"""
    if not events:
        return
    
    event = events[-1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(event, font, 0.8, 2)
    x = (width - tw) // 2 
    y = 80
    
    draw_text_with_bg(frame, event, (x, y), (255, 255, 255), 
                      font_scale=0.8, bg_alpha=0.8)


def draw_token_messages(frame: np.ndarray, messages: Dict[int, str], 
                       width: int, height: int):
    """Draw token hit messages"""
    if not messages:
        return
    
    y_offset = 120
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
        
        # Draw label
        top = get_top_point(card.box)
        label = f"Card {card.card_id}"
        if card.team:
            label += f" ({card.team})"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        draw_text_with_bg(frame, label, (top[0]-tw//2, top[1]-10), 
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
        top = get_top_point(battle.box)
        text = f"Battle: {battle.card1_id}({battle.team1}) vs {battle.card2_id}({battle.team2})"
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        draw_text_with_bg(frame, text, (top[0]-tw//2, top[1]-10), BATTLE_COLOR)


def draw_stats(frame: np.ndarray, frame_count: int, total_frames: int, 
               num_cards: int, num_battles: int, height: int):
    """Draw statistics at the bottom of the frame"""
    stats = f"Frame: {frame_count}/{total_frames} | Cards: {num_cards} | Battles: {num_battles}"
    cv2.putText(frame, stats, (10, height-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def save_detection_visualization(output_dir: Path, frame_number: int,
                                frame: np.ndarray, gray: np.ndarray, 
                                blur: np.ndarray, edges: np.ndarray, 
                                dilated: np.ndarray, mask: np.ndarray, 
                                dist: np.ndarray, fg: np.ndarray, 
                                bg: np.ndarray, unknown: np.ndarray, 
                                markers_post: np.ndarray, detected_cards: List):

    # Normalize distance transform
    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create markers visualization
    markers_color = np.zeros_like(frame)
    num_cards_detected = 0
    for label in np.unique(markers_post):
        if label == -1:
            markers_color[markers_post == -1] = [0, 0, 255]  # RED boundaries
        elif label > 1:
            num_cards_detected += 1
            color = np.random.randint(0, 255, 3).tolist()
            markers_color[markers_post == label] = color
    
    # Create final detection overlay
    final_detection = frame.copy()
    for card in detected_cards:
        # Draw the detected card box
        cv2.drawContours(final_detection, [card.box], -1, (0, 255, 0), 3)
        # Draw center point
        center = (int(card.center[0]), int(card.center[1]))
        cv2.circle(final_detection, center, 5, (0, 0, 255), -1)
    
    # Create 3x3 grid
    row1 = np.hstack([
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    ])
    
    row2 = np.hstack([
        cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        cv2.applyColorMap(dist_vis, cv2.COLORMAP_JET)
    ])
    
    row3 = np.hstack([
        cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(unknown, cv2.COLOR_GRAY2BGR)
    ])
    
    grid = np.vstack([row1, row2, row3])
    
    # Add labels with parameters
    labels = [
        ["1. Grayscale", f"2. Blur 9x9", 
         f"3. Canny 50/150"],
        [f"4. Dilate kernel 3x3", 
         "5. Mask", "6. Distance"],
        [f"7. FG thresh=0.5", 
         f"8. BG iter=2", "9. Unknown"]
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = gray.shape
    for i, row_labels in enumerate(labels):
        for j, label in enumerate(row_labels):
            y = i * h + 25
            x = j * w + 10
            cv2.putText(grid, label, (x, y), font, 0.5, (0, 0, 0), 3)
            cv2.putText(grid, label, (x, y), font, 0.5, (0, 255, 255), 1)
    
    # Add result + final detection
    result_panel = np.hstack([markers_color, frame.copy(), final_detection])
    
    result_labels = ["10. Watershed (RED)", "11. Original", "12. Final Detections"]
    for j, label in enumerate(result_labels):
        x = j * w + 10
        y = 25
        cv2.putText(result_panel, label, (x, y), font, 0.5, (0, 0, 0), 3)
        cv2.putText(result_panel, label, (x, y), font, 0.5, (0, 255, 255), 1)
    
    grid = np.vstack([grid, result_panel])
    
    # Add summary panel
    summary_height = 100
    summary_panel = np.zeros((summary_height, grid.shape[1], 3), dtype=np.uint8)
    
    cv2.putText(summary_panel, "9-Step Card Detection Visualization", 
               (20, 25), font, 0.7, (0, 255, 255), 2)
    cv2.putText(summary_panel, f"Frame {frame_number} | Detected: {num_cards_detected} cards", 
               (20, 50), font, 0.6, (255, 255, 255), 1)
    
    final_grid = np.vstack([grid, summary_panel])
    
    # Save to file
    output_path = output_dir / f"frame_{frame_number:06d}.jpg"
    cv2.imwrite(str(output_path), final_grid)
    return output_path