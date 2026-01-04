#!/usr/bin/env python3
import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque
from card_detector import CardDetector, DetectionParams
from board_detector import BoardDetector

# CONFIGURATION
VIDEO_PATH = "data/hard/hard.mp4"
OUTPUT_DIR = "output_video"
SAVE_VIDEO = True

# Team colors (BGR format for OpenCV)
TEAM_COLORS = {
    'A': (255, 0, 0),    # Blue
    'B': (0, 255, 0)     # Green
}

class GameEventTracker:
    """Track game events like card appearances, movements, battles"""
    def __init__(self):
        self.previous_cards = {}
        self.previous_battles = {}
        self.active_battle_cards = set()
        self.events = deque(maxlen=5)
        self.movement_threshold = 30
        self.battle_winners = {}
        
    def update(self, cards, battles, card_detector):
        """Update state and generate events"""
        current_card_ids = set()
        current_battle_ids = set()
        new_events = []
        
        # Track which cards are currently in battles
        cards_in_battle = set()
        for battle in battles:
            if battle.battle_ids:
                cards_in_battle.update(battle.battle_ids)
        
        # Track current cards (exclude ghosts from events)
        for card in cards:
            if card.is_ghost:
                continue
                
            current_card_ids.add(card.card_id)
            
            # Skip movement tracking for cards in battle
            if card.card_id in cards_in_battle:
                continue
            
            if card.card_id not in self.previous_cards:
                was_in_battle = card.card_id in self.active_battle_cards
                if was_in_battle:
                    team_text = f" (Team {card.team})" if card.team else ""
                    new_events.append(f"Card {card.card_id}{team_text} wins the battle")
                else:
                    team_text = f" (Team {card.team})" if card.team else ""
                    new_events.append(f"Card {card.card_id}{team_text} appears")
            else:
                prev_center = self.previous_cards[card.card_id]
                dist = np.sqrt((card.center[0] - prev_center[0])**2 + 
                             (card.center[1] - prev_center[1])**2)
                if dist > self.movement_threshold:
                    new_events.append(f"Card {card.card_id} moves")
            
            self.previous_cards[card.card_id] = card.center
        
        # Track battles
        for battle in battles:
            if battle.battle_ids:
                tracked_obj = card_detector.tracked_objects.get(battle.card_id)
                
                # Only track confirmed battles
                if tracked_obj and tracked_obj.is_confirmed_battle:
                    current_battle_ids.add(battle.card_id)
                    
                    if battle.card_id not in self.previous_battles:
                        card1, card2 = battle.battle_ids
                        
                        # Get team info for battle participants
                        team1 = card_detector.tracked_objects.get(card1)
                        team2 = card_detector.tracked_objects.get(card2)
                        team1_name = team1.team if team1 else "?"
                        team2_name = team2.team if team2 else "?"
                        
                        new_events.append(f"Battle confirmed: {card1}({team1_name}) vs {card2}({team2_name})")
                        self.previous_battles[battle.card_id] = battle.battle_ids
                        
                        self.active_battle_cards.add(card1)
                        self.active_battle_cards.add(card2)
        
        # Check for battles that ended
        ended_battles = set(self.previous_battles.keys()) - current_battle_ids
        for battle_id in ended_battles:
            card1, card2 = self.previous_battles[battle_id]
            
            surviving_cards = [c for c in cards if c.card_id in [card1, card2] and not c.is_ghost]
            
            if len(surviving_cards) == 1:
                winner = surviving_cards[0].card_id
                loser = card1 if winner == card2 else card2
                
                winner_team = surviving_cards[0].team
                team_text = f" (Team {winner_team})" if winner_team else ""
                new_events.append(f"Card {loser} lost to {winner}{team_text}")
                self.battle_winners[battle_id] = winner
                
                self.active_battle_cards.discard(loser)
                
            elif len(surviving_cards) == 0:
                new_events.append(f"Battle ended (both cards gone)")
                self.active_battle_cards.discard(card1)
                self.active_battle_cards.discard(card2)
            
            del self.previous_battles[battle_id]
        
        # Check for cards that disappeared
        disappeared_cards = set(self.previous_cards.keys()) - current_card_ids
        for card_id in disappeared_cards:
            if card_id not in cards_in_battle and card_id not in self.active_battle_cards:
                new_events.append(f"Card {card_id} disappears")
            
            if card_id not in cards_in_battle:
                del self.previous_cards[card_id]
        
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
        alpha = 1.0 - (i * 0.15)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(event, font, font_scale, thickness)
        
        x_pos = 10
        y_pos = y_offset + (i * 35)
        
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (x_pos - 5, y_pos - text_height - 5),
                     (x_pos + text_width + 10, y_pos + 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
        
        color = tuple(int(c * alpha) for c in [255, 255, 255])
        cv2.putText(frame, event, (x_pos, y_pos),
                   font, font_scale, color, thickness)

def draw_scoreboard(frame, scores, width):
    """Draw team scores at the top of the frame"""
    # Scoreboard background
    board_height = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, board_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    # Team A (Blue) on the left
    team_a_text = f"Team A: {scores['A']}"
    (text_width_a, text_height_a), _ = cv2.getTextSize(team_a_text, font, font_scale, thickness)
    x_a = 50
    y_a = 40
    cv2.putText(frame, team_a_text, (x_a, y_a), font, font_scale, TEAM_COLORS['A'], thickness)
    
    # Team B (Green) on the right
    team_b_text = f"Team B: {scores['B']}"
    (text_width_b, text_height_b), _ = cv2.getTextSize(team_b_text, font, font_scale, thickness)
    x_b = width - text_width_b - 50
    y_b = 40
    cv2.putText(frame, team_b_text, (x_b, y_b), font, font_scale, TEAM_COLORS['B'], thickness)
    
    # Draw divider line
    cv2.line(frame, (width // 2, 0), (width // 2, board_height), (255, 255, 255), 2)

def draw_battle_status(frame, battle, tracked_obj):
    """Draw battle status information"""
    if tracked_obj is None:
        return
    
    status_lines = []
    
    if tracked_obj.frames_both_missing > 0:
        status_lines.append(f"Occluded: {tracked_obj.frames_both_missing}f")
    
    if tracked_obj.frames_one_missing > 0:
        missing_id = tracked_obj.missing_card_id
        remaining = CardDetector().battle_end_threshold - tracked_obj.frames_one_missing
        status_lines.append(f"Card {missing_id} missing: {tracked_obj.frames_one_missing}f")
        if remaining > 0:
            status_lines.append(f"Ending in: {remaining}f")
    
    if status_lines:
        bx, by, bw, bh = tracked_obj.bbox
        status_y = by + bh + 20
        
        for i, line in enumerate(status_lines):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            
            text_x = bx
            text_y = status_y + (i * 20)
            
            cv2.rectangle(frame,
                         (text_x - 3, text_y - text_height - 3),
                         (text_x + text_width + 3, text_y + 3),
                         (0, 0, 0), -1)
            
            cv2.putText(frame, line, (text_x, text_y),
                       font, font_scale, (255, 255, 0), thickness)

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
    
    print("Step 2: Processing frames with improved card detection...")
    print(f"Detection parameters:")
    print(f"  - Card area: {card_detector.params.min_area}-{card_detector.params.max_area}")
    print(f"  - Card aspect: {card_detector.params.min_aspect}-{card_detector.params.max_aspect} (height/width)")
    print(f"  - Battle end threshold: {card_detector.battle_end_threshold} frames")
    print(f"  - Battle noise threshold: {card_detector.battle_noise_threshold} frames")
    print(f"  - Max overlap for new cards: {card_detector.max_new_card_overlap * 100}%")
    
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
            current_board = board_corners
        
        # Detect cards on full frame, then filter to board
        cards, battles = card_detector.detect_cards(frame, current_board)
        
        # Get team scores
        scores = card_detector.get_team_scores(cards, battles)
        
        # Update event tracker
        events = event_tracker.update(cards, battles, card_detector)
        
        # Update statistics (exclude ghosts)
        real_cards = [c for c in cards if not c.is_ghost]
        cards_detected_total += len(real_cards)
        battles_detected_total += len(battles)
        
        # Draw scoreboard
        draw_scoreboard(frame, scores, width)
        
        # Draw board boundary
        if current_board is not None:
            board_int = current_board.astype(np.int32)
            cv2.polylines(frame, [board_int], True, (255, 255, 0), 2)
            
            cv2.putText(frame, "Board", 
                       (board_int[0][0] + 10, board_int[0][1] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw battle pairs with status
        for battle in battles:
            tracked_obj = card_detector.tracked_objects.get(battle.card_id)
            
            # Use different color for unconfirmed battles
            if tracked_obj and not tracked_obj.is_confirmed_battle:
                battle_color = (0, 165, 255)  # Orange for unconfirmed
                cv2.drawContours(frame, [battle.box], 0, battle_color, 2)  # Thinner line
            else:
                battle_color = (0, 0, 255)  # Red for confirmed
                cv2.drawContours(frame, [battle.box], 0, battle_color, 3)
            
            top_point = get_top_point(battle.box)
            
            if battle.battle_ids:
                # Get team info for battle participants
                team1 = card_detector.tracked_objects.get(battle.battle_ids[0])
                team2 = card_detector.tracked_objects.get(battle.battle_ids[1])
                team1_name = team1.team if team1 else "?"
                team2_name = team2.team if team2 else "?"
                
                if tracked_obj and not tracked_obj.is_confirmed_battle:
                    conf_count = tracked_obj.battle_confirmation_count
                    conf_needed = card_detector.params.battle_confirmation_frames
                    text = f"Battle? {battle.battle_ids[0]}({team1_name}) vs {battle.battle_ids[1]}({team2_name}) [{conf_count}/{conf_needed}]"
                else:
                    text = f"Battle {battle.battle_ids[0]}({team1_name}) vs {battle.battle_ids[1]}({team2_name})"
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
                       font, font_scale, battle_color, thickness)
            
            # Draw battle status (only for confirmed battles)
            if tracked_obj and tracked_obj.is_confirmed_battle:
                draw_battle_status(frame, battle, tracked_obj)

        # Draw individual cards with team colors
        for card in cards:
            # Get team color
            color = TEAM_COLORS.get(card.team, (0, 255, 0))
            
            # Draw with transparency if ghost
            if card.is_ghost:
                overlay = frame.copy()
                cv2.drawContours(overlay, [card.box], 0, (128, 128, 128), 2)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            else:
                cv2.drawContours(frame, [card.box], 0, color, 3)
            
            top_point = get_top_point(card.box)
            team_text = f" ({card.team})" if card.team else ""
            label = f"Card {card.card_id}{team_text}"
            if card.is_ghost:
                label += " [ghost]"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            text_x = top_point[0] - text_width // 2
            text_y = top_point[1] - 10
            
            bg_alpha = 0.5 if card.is_ghost else 1.0
            overlay = frame.copy()
            cv2.rectangle(overlay,
                         (text_x - 3, text_y - text_height - 3),
                         (text_x + text_width + 3, text_y + 3),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
            
            label_color = (128, 128, 128) if card.is_ghost else color
            cv2.putText(frame, label, (text_x, text_y), 
                       font, font_scale, label_color, thickness)

        # Draw events
        draw_events(frame, events, height)

        # Add frame statistics
        stats_text = f"Frame: {frame_count}/{total_frames} | Cards: {len(real_cards)} | Battles: {len(battles)}"
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
                  f"Avg Cards: {avg_cards:.1f} | Avg Battles: {avg_battles:.1f} | "
                  f"Score A:{scores['A']} B:{scores['B']}")

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
    print(f"Final Score - Team A: {scores['A']}, Team B: {scores['B']}")
    print(f"Output saved to: {output_dir / f'detected_{video_path.stem}.mp4'}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()