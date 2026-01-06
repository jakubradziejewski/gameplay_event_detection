import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque
from card_detector import CardDetector
from board_detector import BoardDetector
from token_detector import TokenDetector
from dice_detector import DiceDetector
from visualization import (
    draw_dice, draw_scoreboard, draw_board, draw_cards, draw_battles,
    draw_events, draw_token_messages, draw_stats
)

# Configuration
VIDEO_PATH = "data/medium/medium3.mp4"
OUTPUT_DIR = "output_video_tokens"

class GameEventTracker:
    def __init__(self):
        self.previous_cards = {}
        self.previous_battles = set()
        self.recent_battle_losers = {}
        self.current_frame = 0
        self.events = deque(maxlen=5)
        self.loser_cooldown_frames = 10
        
    def update(self, cards, battles, detector):
        self.current_frame += 1
        current_cards = set()
        current_battles = set()
        new_events = []
        
        # Track current battles
        for battle in battles:
            current_battles.add(battle.battle_id)
            
            # New battle started
            if battle.battle_id not in self.previous_battles:
                new_events.append(
                    f"Battle: Card {battle.card1_id} (Team {battle.team1}) vs "
                    f"Card {battle.card2_id} (Team {battle.team2})"
                )
        
        # Check for battle loss events from detector
        if hasattr(detector, 'battle_events') and detector.battle_events:
            new_events.extend(detector.battle_events)
            # Extract card IDs that lost battles and mark them
            for event in detector.battle_events:
                if "lost the battle" in event:
                    parts = event.split()
                    if len(parts) > 1:
                        try:
                            card_id = int(parts[1])
                            self.recent_battle_losers[card_id] = self.current_frame
                        except ValueError:
                            pass
        
        # Clean up old battle losers
        expired_losers = [
            card_id for card_id, lost_frame in list(self.recent_battle_losers.items())
            if self.current_frame - lost_frame > self.loser_cooldown_frames
        ]
        for card_id in expired_losers:
            del self.recent_battle_losers[card_id]
        
        # Track active battle cards
        active_battle_cards = set()
        for battle in battles:
            active_battle_cards.add(battle.card1_id)
            active_battle_cards.add(battle.card2_id)
        
        # Track card appearances and movements
        for card in cards:
            current_cards.add(card.card_id)
            
            # Skip cards in battles or recently lost
            if card.card_id in active_battle_cards or card.card_id in self.recent_battle_losers:
                continue
                
            if card.card_id not in self.previous_cards:
                new_events.append(f"Card {card.card_id} (Team {card.team}) appears")
            else:
                prev = self.previous_cards[card.card_id]
                dist = np.sqrt((card.center[0]-prev[0])**2 + (card.center[1]-prev[1])**2)
                if dist > 30:
                    new_events.append(f"Card {card.card_id} (Team {card.team}) moves")
            
            self.previous_cards[card.card_id] = card.center
        
        # Clean up tracking for disappeared cards
        disappeared_cards = set(self.previous_cards.keys()) - current_cards
        for cid in disappeared_cards:
            del self.previous_cards[cid]
        
        # Update battle tracking
        self.previous_battles = current_battles
        
        # Add new events to queue
        for event in new_events:
            self.events.append(event)
        
        return list(self.events)

def main():
    video_path = Path(VIDEO_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {total} frames...")
    
    # Initialize components
    board_detector = BoardDetector(str(video_path))
    ret, first = cap.read()
    if not ret:
        return
    board = board_detector.calibrate(first)
    print(f"Board detected: {board.astype(int).tolist() if board is not None else 'Full frame'}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    out = cv2.VideoWriter(str(output_dir / f"detected_{video_path.stem}.mp4"),
                         cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Initialize detector with visualization enabled
    detector = CardDetector(enable_visualization=True)
    tracker = GameEventTracker()
    token_detector = TokenDetector()
    dice_detector = DiceDetector(history_length=20, distance_threshold=30, dice_radius=40)

    print(f"Dice detection: history_length={dice_detector.history_length}, "
          f"distance_threshold={dice_detector.distance_threshold}, "
          f"dice_radius={dice_detector.dice_radius}")
    
    start = time.time()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect board
        curr_board = board_detector._detect_inner_from_brightness(frame)
        curr_board = board_detector.smooth_detection(curr_board) if curr_board is not None else board
        
        # Detect cards and battles
        cards, battles = detector.detect_cards(frame, curr_board)
        scores = detector.get_team_scores(cards, battles)
        events = tracker.update(cards, battles, detector)
        
        # Detect tokens for active battles
        if battles:
            battle_tokens = token_detector.detect_tokens_for_battles(frame, battles, curr_board)
            token_detector.update_battle_tokens(battle_tokens, battles, detector.tracked_objects, frame)
        
        # Get token messages
        token_messages = token_detector.get_battle_messages(battles)

        # Detect dice if board is available
        dice_list = []
        if curr_board is not None:
            dice_list = dice_detector.detect_dice(frame, curr_board)

        # Draw everything using visualization functions
        draw_dice(frame, dice_list)
        draw_scoreboard(frame, scores, width)
        draw_board(frame, curr_board)
        draw_battles(frame, battles, token_detector)
        draw_cards(frame, cards)
        draw_events(frame, events, width)
        draw_token_messages(frame, token_messages, width, height)
        draw_stats(frame, frame_count, total, len(cards), len(battles), height)

        out.write(frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            elapsed = time.time() - start
            print(f"Frame {frame_count}/{total} | Speed: {frame_count/elapsed:.2f} fps | "
                  f"Score A:{scores['A']} B:{scores['B']} | "
                  f"Dice: {len(dice_list)}")
    
    cap.release()
    out.release()
    
    elapsed = time.time() - start

    print("\nSummary:")
    print(f"Frames: {frame_count} | Time: {elapsed:.2f}s ({frame_count/elapsed:.2f} fps)")
    print(f"Final Score - Team A: {scores['A']}, Team B: {scores['B']}")
    print(f"Output: {output_dir / f'detected_{video_path.stem}.mp4'}")


if __name__ == "__main__":
    main()