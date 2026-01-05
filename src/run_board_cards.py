import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque
from card_detector import CardDetector
from board_detector import BoardDetector

# Configuration
VIDEO_PATH = "data/medium/medium3.mp4"
OUTPUT_DIR = "output_video"
TEAM_COLORS = {'A': (255, 0, 0), 'B': (0, 255, 0)}

class GameEventTracker:
    def __init__(self):
        self.previous_cards = {}
        self.previous_battles = set()  # Just track battle IDs
        self.cards_that_lost_battle = set()  # Track cards that just lost a battle
        self.events = deque(maxlen=5)
        
    def update(self, cards, battles, detector):
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
            # Extract card IDs that lost battles
            for event in detector.battle_events:
                if "lost the battle" in event:
                    # Extract card ID from "Card X (Team Y) lost the battle"
                    parts = event.split()
                    if len(parts) > 1:
                        try:
                            card_id = int(parts[1])
                            self.cards_that_lost_battle.add(card_id)
                        except ValueError:
                            pass
        
        # Track active battle cards to exclude from movement tracking
        active_battle_cards = set()
        for battle in battles:
            active_battle_cards.add(battle.card1_id)
            active_battle_cards.add(battle.card2_id)
        
        # Track card appearances and movements (excluding cards in battles)
        for card in cards:
            if card.is_ghost:
                continue
            
            current_cards.add(card.card_id)
            
            # Skip cards that are currently in battles OR just lost a battle
            if card.card_id in active_battle_cards or card.card_id in self.cards_that_lost_battle:
                continue
                
            if card.card_id not in self.previous_cards:
                # New card appeared
                new_events.append(f"Card {card.card_id} (Team {card.team}) appears")
            else:
                # Check if card moved significantly
                prev = self.previous_cards[card.card_id]
                dist = np.sqrt((card.center[0]-prev[0])**2 + (card.center[1]-prev[1])**2)
                if dist > 30:
                    new_events.append(f"Card {card.card_id} (Team {card.team}) moves")
            
            self.previous_cards[card.card_id] = card.center
        
        # Clear the cards_that_lost_battle set (only skip movement for one frame)
        self.cards_that_lost_battle.clear()
        
        # Clean up tracking for cards that are no longer visible
        disappeared_cards = set(self.previous_cards.keys()) - current_cards
        for cid in disappeared_cards:
            del self.previous_cards[cid]
        
        # Update battle tracking
        self.previous_battles = current_battles
        
        # Add new events to the queue
        for event in new_events:
            self.events.append(event)
        
        return list(self.events)

def draw_text_with_bg(frame, text, pos, color, font_scale=0.7, thickness=2, bg_alpha=0.7):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    overlay = frame.copy()
    cv2.rectangle(overlay, (x-5, y-th-5), (x+tw+5, y+5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1-bg_alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def draw_events(frame, events, width):
    if not events:
        return
    event = events[-1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(event, font, 0.8, 2)
    x = (width - tw) // 2 
    y = 80
    draw_text_with_bg(frame, event, (x, y), (255, 255, 255), font_scale=0.8, bg_alpha=0.8)

def draw_scoreboard(frame, scores, width):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Team A: {scores['A']}", (50, 40), font, 1.2, TEAM_COLORS['A'], 3)
    (tw, _), _ = cv2.getTextSize(f"Team B: {scores['B']}", font, 1.2, 3)
    cv2.putText(frame, f"Team B: {scores['B']}", (width-tw-50, 40), font, 1.2, TEAM_COLORS['B'], 3)
    cv2.line(frame, (width//2, 0), (width//2, 60), (255, 255, 255), 2)

def get_top_point(box):
    return box[np.argmin(box[:, 1])]

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
    board_detector = BoardDetector(str(video_path))
    ret, first = cap.read()
    if not ret:
        return
    board = board_detector.calibrate(first)
    print(f"Board detected: {board.astype(int).tolist() if board is not None else 'Full frame'}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    out = cv2.VideoWriter(str(output_dir / f"detected_{video_path.stem}.mp4"),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    detector = CardDetector()
    tracker = GameEventTracker()

    print(f"Parameters: Card area={detector.params.min_area}-{detector.params.max_area}, "
          f"Aspect={detector.params.min_aspect:.1f}-{detector.params.max_aspect:.1f}, "
          f"Battle buffer={detector.params.battle_proximity_buffer*100:.0f}%")
    
    start = time.time()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_board = board_detector._detect_inner_from_brightness(frame)
        curr_board = board_detector.smooth_detection(curr_board) if curr_board is not None else board
        
        cards, battles = detector.detect_cards(frame, curr_board)
        scores = detector.get_team_scores(cards, battles)
        events = tracker.update(cards, battles, detector)
        
        # Filter out ghost cards for rendering
        real_cards = [c for c in cards if not c.is_ghost]
        
        draw_scoreboard(frame, scores, width)
        
        if curr_board is not None:
            cv2.polylines(frame, [curr_board.astype(np.int32)], True, (255, 255, 0), 2)
            cv2.putText(frame, "Board", (int(curr_board[0][0]+10), int(curr_board[0][1]+30)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw battles
        for battle in battles:
            color = (0, 0, 255)
            cv2.drawContours(frame, [battle.box], 0, color, 3)
            
            top = get_top_point(battle.box)
            text = f"Battle: {battle.card1_id}({battle.team1}) vs {battle.card2_id}({battle.team2})"
            
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            draw_text_with_bg(frame, text, (top[0]-tw//2, top[1]-10), color)

        # Draw cards
        for card in real_cards:
            color = TEAM_COLORS.get(card.team, (0, 255, 0))
            cv2.drawContours(frame, [card.box], 0, color, 2 if card.in_edge_zone else 3)
            
            top = get_top_point(card.box)
            label = f"Card {card.card_id}"
            if card.team:
                label += f" ({card.team})"
            if card.in_edge_zone:
                label += " [edge]"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            draw_text_with_bg(frame, label, (top[0]-tw//2, top[1]-10), color, font_scale=0.6)

        draw_events(frame, events, width)
        
        stats = f"Frame: {frame_count}/{total} | Cards: {len(real_cards)} | Battles: {len(battles)}"
        cv2.putText(frame, stats, (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

        frame_count += 1
        if frame_count % 50 == 0:
            elapsed = time.time() - start
            print(f"Frame {frame_count}/{total} | Speed: {frame_count/elapsed:.2f} fps | "
                  f"Score A:{scores['A']} B:{scores['B']} | Active battles: {len(battles)}")
    
    cap.release()
    out.release()
    
    elapsed = time.time() - start

    print("\nSummary:")
    print(f"Frames: {frame_count} | Time: {elapsed:.2f}s ({frame_count/elapsed:.2f} fps)")
    print(f"Final Score - Team A: {scores['A']}, Team B: {scores['B']}")
    print(f"Output: {output_dir / f'detected_{video_path.stem}.mp4'}")


if __name__ == "__main__":
    main()