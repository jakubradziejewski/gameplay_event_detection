#!/usr/bin/env python3
import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque
from card_detector import CardDetector
from board_detector import BoardDetector

# Configuration
VIDEO_PATH = "data/medium/medium.mp4"
OUTPUT_DIR = "output_video"
SAVE_VIDEO = True
TEAM_COLORS = {'A': (255, 0, 0), 'B': (0, 255, 0)}

class GameEventTracker:
    def __init__(self):
        self.previous_cards = {}
        self.previous_battles = {}
        self.cards_in_battles = {}
        self.events = deque(maxlen=5)
        
    def update(self, cards, battles, detector):
        current_cards = set()
        current_battles = set()
        current_battle_cards = set()
        current_battle_map = {}
        new_events = []
        
        # Track confirmed battles
        for battle in battles:
            if battle.battle_ids:
                tracked = detector.tracked_objects.get(battle.card_id)
                if tracked and tracked.is_confirmed_battle:
                    current_battles.add(battle.card_id)
                    for cid in battle.battle_ids:
                        current_battle_cards.add(cid)
                        current_battle_map[cid] = battle.card_id
                    
                    if battle.card_id not in self.previous_battles:
                        c1, c2 = battle.battle_ids
                        t1 = detector.tracked_objects.get(c1)
                        t2 = detector.tracked_objects.get(c2)
                        new_events.append(f"Battle: {c1}({t1.team if t1 else '?'}) vs {c2}({t2.team if t2 else '?'})")
                        self.previous_battles[battle.card_id] = battle.battle_ids
                        self.cards_in_battles[c1] = battle.card_id
                        self.cards_in_battles[c2] = battle.card_id
        
        # Check ended battles
        for bid in set(self.previous_battles.keys()) - current_battles:
            c1, c2 = self.previous_battles[bid]
            real_survivors = [c for c in cards if c.card_id in [c1, c2] and not c.is_ghost]
            
            if len(real_survivors) == 1:
                winner = real_survivors[0]
                loser = c1 if winner.card_id == c2 else c2
                new_events.append(f"Card {loser} lost to {winner.card_id} (Team {winner.team})")
                self.cards_in_battles.pop(loser, None)
                self.cards_in_battles.pop(winner.card_id, None)
            elif len(real_survivors) == 0:
                new_events.append(f"Battle cancelled")
                self.cards_in_battles.pop(c1, None)
                self.cards_in_battles.pop(c2, None)
            del self.previous_battles[bid]
        
        # Track cards
        for card in cards:
            if card.is_ghost:
                continue
            current_cards.add(card.card_id)
            if card.card_id in current_battle_cards:
                continue
            
            was_in_battle = card.card_id in self.cards_in_battles
            if card.card_id not in self.previous_cards:
                if was_in_battle:
                    new_events.append(f"Card {card.card_id} (Team {card.team}) wins")
                    del self.cards_in_battles[card.card_id]
                else:
                    new_events.append(f"Card {card.card_id} (Team {card.team}) appears")
            else:
                prev = self.previous_cards[card.card_id]
                dist = np.sqrt((card.center[0]-prev[0])**2 + (card.center[1]-prev[1])**2)
                if dist > 30:
                    if was_in_battle:
                        new_events.append(f"Card {card.card_id} fled")
                        del self.cards_in_battles[card.card_id]
                    else:
                        new_events.append(f"Card {card.card_id} moves")
            self.previous_cards[card.card_id] = card.center
        
        # Check disappeared cards
        for cid in set(self.previous_cards.keys()) - current_cards:
            if cid in current_battle_cards:
                continue
            if cid in self.cards_in_battles:
                new_events.append(f"Card {cid} lost (removed)")
                del self.cards_in_battles[cid]
            else:
                new_events.append(f"Card {cid} disappears")
            del self.previous_cards[cid]
        
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

def draw_events(frame, events):
    for i, event in enumerate(reversed(events)):
        alpha = 1.0 - (i * 0.15)
        color = tuple(int(c * alpha) for c in [255, 255, 255])
        draw_text_with_bg(frame, event, (10, 50 + i*35), color, bg_alpha=alpha*0.7)

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
    print("Step 1: Calibrating board...")
    
    board_detector = BoardDetector(str(video_path))
    ret, first = cap.read()
    if not ret:
        return
    board = board_detector.calibrate(first)
    print(f"✓ Board detected: {board.astype(int).tolist() if board is not None else 'Full frame'}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if SAVE_VIDEO:
        out = cv2.VideoWriter(str(output_dir / f"detected_{video_path.stem}.mp4"),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    detector = CardDetector()
    tracker = GameEventTracker()
    
    print("Step 2: Processing frames...")
    print(f"Parameters: Card area={detector.params.min_area}-{detector.params.max_area}, "
          f"Aspect={detector.params.min_aspect:.1f}-{detector.params.max_aspect:.1f}, "
          f"Edge buffer X={detector.params.edge_buffer_x*100:.0f}% Y={detector.params.edge_buffer_y*100:.0f}%")
    
    start = time.time()
    frame_count = total_cards = total_battles = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_board = board_detector._detect_inner_from_brightness(frame)
        curr_board = board_detector.smooth_detection(curr_board) if curr_board is not None else board
        
        cards, battles = detector.detect_cards(frame, curr_board)
        scores = detector.get_team_scores(cards, battles)
        events = tracker.update(cards, battles, detector)
        
        real_cards = [c for c in cards if not c.is_ghost]
        total_cards += len(real_cards)
        total_battles += len(battles)
        
        draw_scoreboard(frame, scores, width)
        
        if curr_board is not None:
            cv2.polylines(frame, [curr_board.astype(np.int32)], True, (255, 255, 0), 2)
            cv2.putText(frame, "Board", (int(curr_board[0][0]+10), int(curr_board[0][1]+30)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw battles
        for battle in battles:
            tracked = detector.tracked_objects.get(battle.card_id)
            confirmed = tracked and tracked.is_confirmed_battle
            color = (0, 0, 255) if confirmed else (0, 165, 255)
            cv2.drawContours(frame, [battle.box], 0, color, 3 if confirmed else 2)
            
            top = get_top_point(battle.box)
            if battle.battle_ids:
                t1 = detector.tracked_objects.get(battle.battle_ids[0])
                t2 = detector.tracked_objects.get(battle.battle_ids[1])
                team1 = t1.team if t1 else "?"
                team2 = t2.team if t2 else "?"
                if not confirmed:
                    cnt = tracked.battle_confirmation_count
                    text = f"Battle? {battle.battle_ids[0]}({team1}) vs {battle.battle_ids[1]}({team2}) [{cnt}/{detector.params.battle_confirmation_frames}]"
                else:
                    text = f"Battle {battle.battle_ids[0]}({team1}) vs {battle.battle_ids[1]}({team2})"
            else:
                text = "Battle"
            
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            draw_text_with_bg(frame, text, (top[0]-tw//2, top[1]-10), color)

        # Draw cards
        for card in cards:
            color = TEAM_COLORS.get(card.team, (0, 255, 0))
            if card.is_ghost:
                overlay = frame.copy()
                cv2.drawContours(overlay, [card.box], 0, (128, 128, 128), 2)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            else:
                cv2.drawContours(frame, [card.box], 0, color, 2 if card.in_edge_zone else 3)
            
            top = get_top_point(card.box)
            label = f"Card {card.card_id}"
            if card.team:
                label += f" ({card.team})"
            if card.is_ghost:
                label += " [ghost]"
            elif card.in_edge_zone:
                label += " [edge]"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lcolor = (128, 128, 128) if card.is_ghost else color
            draw_text_with_bg(frame, label, (top[0]-tw//2, top[1]-10), lcolor,
                            font_scale=0.6, bg_alpha=0.5 if card.is_ghost else 1.0)

        draw_events(frame, events)
        
        stats = f"Frame: {frame_count}/{total} | Cards: {len(real_cards)} | Battles: {len(battles)}"
        cv2.putText(frame, stats, (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if SAVE_VIDEO:
            out.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start
            print(f"Frame {frame_count}/{total} | Speed: {frame_count/elapsed:.2f} fps | "
                  f"Avg Cards: {total_cards/frame_count:.1f} | Battles: {total_battles/frame_count:.1f} | "
                  f"Score A:{scores['A']} B:{scores['B']}")

    cap.release()
    if SAVE_VIDEO:
        out.release()
    
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Frames: {frame_count} | Time: {elapsed:.2f}s ({frame_count/elapsed:.2f} fps)")
    print(f"Avg Cards: {total_cards/frame_count:.2f} | Avg Battles: {total_battles/frame_count:.2f}")
    print(f"Final Score - Team A: {scores['A']}, Team B: {scores['B']}")
    print(f"Output: {output_dir / f'detected_{video_path.stem}.mp4'}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()