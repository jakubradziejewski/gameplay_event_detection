#!/usr/bin/env python3
import cv2
import time
import numpy as np
from pathlib import Path
from card_detector import CardDetector, DetectionParams

# CONFIGURATION
VIDEO_PATH = "data/easy/game_easy.mp4"
OUTPUT_DIR = "output_video"
SAVE_VIDEO = True

def get_top_point(box):
    """Get the topmost point of a bounding box for label placement"""
    # Find the point with minimum y-coordinate (top of image)
    top_idx = np.argmin(box[:, 1])
    return box[top_idx]

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

    # Initialize Video Writer
    if SAVE_VIDEO:
        out_name = output_dir / f"detected_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_name), fourcc, fps, (width, height))

    detector = CardDetector()
    
    print(f"Processing {total_frames} frames...")
    start_time = time.time()

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process Frame
        cards, battles = detector.detect_cards(frame)

        # Draw battle pairs
        for battle in battles:
            # Draw red box for battle
            cv2.drawContours(frame, [battle.box], 0, (0, 0, 255), 3)
            
            # Get top point for label
            top_point = get_top_point(battle.box)
            text = "Battle"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Position text above the top point
            text_x = top_point[0] - text_width // 2
            text_y = top_point[1] - 10
            
            # Add background rectangle for better visibility
            cv2.rectangle(frame,
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + 5),
                         (0, 0, 0), -1)
            
            cv2.putText(frame, text, (text_x, text_y), 
                       font, font_scale, (0, 0, 255), thickness)

        # Draw individual cards
        for card in cards:
            # Draw green box for normal cards
            cv2.drawContours(frame, [card.box], 0, (0, 255, 0), 2)
            
            # Add card ID label at the top of the card
            top_point = get_top_point(card.box)
            label = f"Card {card.card_id}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position text above the top point
            text_x = top_point[0] - text_width // 2
            text_y = top_point[1] - 10
            
            # Add background rectangle for better visibility
            cv2.rectangle(frame,
                         (text_x - 3, text_y - text_height - 3),
                         (text_x + text_width + 3, text_y + 3),
                         (0, 0, 0), -1)
            
            # Draw the text
            cv2.putText(frame, label, (text_x, text_y), 
                       font, font_scale, (0, 255, 0), thickness)

        if SAVE_VIDEO:
            writer.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Frame {frame_count}/{total_frames} | Speed: {frame_count/elapsed:.2f} fps")

    cap.release()
    if SAVE_VIDEO:
        writer.release()
    
    print(f"\nFinished! Output saved to: {output_dir}")

if __name__ == "__main__":
    main()