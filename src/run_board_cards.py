#!/usr/bin/env python3
import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque
from card_detector import CardDetector, DetectionParams
from board_detector import BoardDetector

# CONFIGURATION
VIDEO_PATH = "data/easy/game_easy.mp4"
OUTPUT_DIR = "output_video"
SAVE_VIDEO = True
BRIGHTNESS_THRESHOLD = 200

class GameEventTracker:
    """Track game events like card appearances, movements, battles"""
    def __init__(self):
        self.previous_cards = {}  # {card_id: center}
        self.previous_battles = {}  # {battle_id: (card1_id, card2_id)}
        self.events = deque(maxlen=5)  # Keep last 5 events
        self.movement_threshold = 30  # Pixels to consider as movement
        
    def update(self, cards, battles):
        """Update state and generate events"""
        current_card_ids = set()
        current_battle_ids = set()
        new_events = []
        
        # Track current cards
        for card in cards:
            current_card_ids.add(card.card_id)
            
            if card.card_id not in self.previous_cards:
                # New card appeared
                new_events.append(f"Card {card.card_id} appears")
            else:
                # Check if card moved
                prev_center = self.previous_cards[card.card_id]
                dist = np.sqrt((card.center[0] - prev_center[0])**2 + 
                             (card.center[1] - prev_center[1])**2)
                if dist > self.movement_threshold:
                    new_events.append(f"Card {card.card_id} moves")
            
            self.previous_cards[card.card_id] = card.center
        
        # Track battles
        for battle in battles:
            if battle.battle_ids:
                current_battle_ids.add(battle.card_id)
                
                if battle.card_id not in self.previous_battles:
                    # New battle started
                    card1, card2 = battle.battle_ids
                    new_events.append(f"Card {card1} and {card2} have battle")
                    self.previous_battles[battle.card_id] = battle.battle_ids
        
        # Check for cards that disappeared
        disappeared_cards = set(self.previous_cards.keys()) - current_card_ids
        for card_id in disappeared_cards:
            # Check if it's part of a battle now
            is_in_battle = False
            for battle in battles:
                if battle.battle_ids and card_id in battle.battle_ids:
                    is_in_battle = True
                    break
            
            if not is_in_battle:
                new_events.append(f"Card {card_id} disappears")
            
            del self.previous_cards[card_id]
        
        # Check for battles that ended
        ended_battles = set(self.previous_battles.keys()) - current_battle_ids
        for battle_id in ended_battles:
            card1, card2 = self.previous_battles[battle_id]
            
            # Check which card survived
            surviving_cards = [c for c in cards if c.card_id in [card1, card2]]
            
            if len(surviving_cards) == 1:
                winner = surviving_cards[0].card_id
                loser = card1 if winner == card2 else card2
                new_events.append(f"Card {loser} lost the battle")
            elif len(surviving_cards) == 0:
                # Both disappeared (rare case)
                new_events.append(f"Battle ended")
            
            del self.previous_battles[battle_id]
        
        # Add new events to queue
        for event in new_events:
            self.events.append(event)
        
        return list(self.events)

def get_top_point(box):
    """Get the topmost point of a bounding box for label placement"""
    top_idx = np.argmin(box[:, 1])
    return box[top_idx]

def draw_events(frame, events, height):
    """Draw event messages on frame"""
    y_offset = 50
    for i, event in enumerate(reversed(events)):
        alpha = 1.0 - (i * 0.15)  # Fade older events
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(event, font, font_scale, thickness)
        
        # Draw background
        x_pos = 10
        y_pos = y_offset + (i * 35)
        
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (x_pos - 5, y_pos - text_height - 5),
                     (x_pos + text_width + 10, y_pos + 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
        
        # Draw text
        color = tuple(int(c * alpha) for c in [255, 255, 255])
        cv2.putText(frame, event, (x_pos, y_pos),
                   font, font_scale, color, thickness)

def main():
    video_path = Path(VIDEO_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # Video Properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {total_frames} frames...")
    print("Step 1: Calibrating board detection...")
    
    # Initialize board detector
    board_detector = BoardDetector(str(video_path))
    
    # Calibrate with first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    board_corners = board_detector.calibrate(first_frame)
    
    if board_corners is None:
        print("Warning: Could not detect board. Processing entire frame.")
    else:
        print(f"✓ Board detected: {board_corners.astype(int).tolist()}")
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Initialize Video Writer
    if SAVE_VIDEO:
        out_name = output_dir / f"detected_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_name), fourcc, fps, (width, height))

    # Initialize card detector and event tracker
    card_detector = CardDetector()
    event_tracker = GameEventTracker()
    
    print("Step 2: Processing frames with card detection...")
    start_time = time.time()

    frame_count = 0
    cards_detected_total = 0
    battles_detected_total = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect board in current frame (with smoothing)
        current_board = board_detector._detect_inner_from_brightness(frame)
        if current_board is not None:
            current_board = board_detector.smooth_detection(current_board)
        else:
            # Use last known board position
            current_board = board_corners
        
        # Detect cards on full frame, then filter to board
        cards, battles = card_detector.detect_cards(frame, current_board)
        
        # Update event tracker
        events = event_tracker.update(cards, battles)
        
        # Update statistics
        cards_detected_total += len(cards)
        battles_detected_total += len(battles)
        
        # Draw board boundary
        if current_board is not None:
            board_int = current_board.astype(np.int32)
            cv2.polylines(frame, [board_int], True, (255, 255, 0), 2)
            
            # Add board label
            cv2.putText(frame, "Board", 
                       (board_int[0][0] + 10, board_int[0][1] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw battle pairs
        for battle in battles:
            cv2.drawContours(frame, [battle.box], 0, (0, 0, 255), 3)
            
            top_point = get_top_point(battle.box)
            
            # Create battle label with IDs if available
            if battle.battle_ids:
                text = f"Battle {battle.battle_ids[0]} vs {battle.battle_ids[1]}"
            else:
                text = f"Battle"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            text_x = top_point[0] - text_width // 2
            text_y = top_point[1] - 10
            
            cv2.rectangle(frame,
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + 5),
                         (0, 0, 0), -1)
            
            cv2.putText(frame, text, (text_x, text_y), 
                       font, font_scale, (0, 0, 255), thickness)

        # Draw individual cards
        for card in cards:
            cv2.drawContours(frame, [card.box], 0, (0, 255, 0), 2)
            
            top_point = get_top_point(card.box)
            label = f"Card {card.card_id}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            text_x = top_point[0] - text_width // 2
            text_y = top_point[1] - 10
            
            cv2.rectangle(frame,
                         (text_x - 3, text_y - text_height - 3),
                         (text_x + text_width + 3, text_y + 3),
                         (0, 0, 0), -1)
            
            cv2.putText(frame, label, (text_x, text_y), 
                       font, font_scale, (0, 255, 0), thickness)

        # Draw events
        draw_events(frame, events, height)

        # Add frame statistics
        stats_text = f"Frame: {frame_count}/{total_frames} | Cards: {len(cards)} | Battles: {len(battles)}"
        cv2.putText(frame, stats_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if SAVE_VIDEO:
            writer.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            avg_cards = cards_detected_total / frame_count
            avg_battles = battles_detected_total / frame_count
            print(f"Frame {frame_count}/{total_frames} | "
                  f"Speed: {frame_count/elapsed:.2f} fps | "
                  f"Avg Cards: {avg_cards:.1f} | Avg Battles: {avg_battles:.1f}")

    cap.release()
    if SAVE_VIDEO:
        writer.release()
    
    elapsed_total = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total frames processed: {frame_count}")
    print(f"Processing time: {elapsed_total:.2f}s ({frame_count/elapsed_total:.2f} fps)")
    print(f"Average cards per frame: {cards_detected_total/frame_count:.2f}")
    print(f"Average battles per frame: {battles_detected_total/frame_count:.2f}")
    print(f"Output saved to: {output_dir / f'detected_{video_path.stem}.mp4'}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()