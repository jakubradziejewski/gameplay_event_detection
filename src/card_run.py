#!/usr/bin/env python3
import cv2
import time
from pathlib import Path
from card_detector import CardDetector, DetectionParams

# CONFIGURATION
VIDEO_PATH = "data/medium/game_medium.mp4"
OUTPUT_DIR = "output_video"
SAVE_VIDEO = True

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
        boxes = detector.detect_cards(frame)

        # Draw results
        for box in boxes:
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            # Optional: label the card
            cv2.putText(frame, "Card", (box[0][0], box[0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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