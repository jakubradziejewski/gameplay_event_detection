#!/usr/bin/env python3
"""
Main runner for Summoner Wars board detection
Configure all settings as variables below
"""

import cv2
from pathlib import Path
from board_detector import BoardDetector
from visualization_utils import VisualizationUtils

# Input video path
VIDEO_PATH = "data/easy/game_video.mp4"

# Output settings
OUTPUT_DIR = "output"
OUTPUT_VIDEO_NAME = None  # None = auto-generate, or specify like "my_output.mp4"

# Visualization flags
VISUALIZE = True              # Create output video with detection overlay
SHOW_LABELS = False           # Show cell labels (A1-H6) on grid
SAVE_DETECTION_STEPS = True  # Save step-by-step detection PNG
SAVE_FILTER_ANALYSIS = True  # Save filter analysis PNG

# Detection parameters
BRIGHTNESS_THRESHOLD = 180    # Brightness threshold (lower for darker boards: 120-150)
GRID_COLS = 8                 # Number of grid columns
GRID_ROWS = 6                 # Number of grid rows



def main():
    # Validate input video
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    # Set up output paths
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if OUTPUT_VIDEO_NAME is None and VISUALIZE:
        output_video = str(output_dir / f"board_detection_{video_path.stem}.mp4")
    elif OUTPUT_VIDEO_NAME is not None:
        output_video = str(output_dir / OUTPUT_VIDEO_NAME)
    else:
        output_video = None
    
    # Print configuration
    print("=" * 70)
    print("SUMMONER WARS BOARD DETECTION")
    print("=" * 70)
    print(f"Input video:        {video_path}")
    print(f"Output directory:   {output_dir}")
    if VISUALIZE:
        print(f"Output video:       {output_video}")
    print(f"Brightness thresh:  {BRIGHTNESS_THRESHOLD}")
    print(f"Grid size:          {GRID_COLS}x{GRID_ROWS}")
    print(f"Show labels:        {SHOW_LABELS}")
    print(f"Save steps viz:     {SAVE_DETECTION_STEPS}")
    print(f"Save filter viz:    {SAVE_FILTER_ANALYSIS}")
    print("=" * 70)
    print()
    
    # Load first frame for analysis
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Failed to open video: {video_path}")
        return 1
    
    ret, first_frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Failed to read first frame")
        return 1
    
    # Save filter analysis if requested
    if SAVE_FILTER_ANALYSIS:
        print("\nGenerating filter analysis...")
        filter_output = output_dir / "filter_analysis.png"
        VisualizationUtils.save_filter_comparison(first_frame, str(filter_output))
        print()
    
    # Initialize detector
    detector = BoardDetector(str(video_path))
    detector.grid_cols = GRID_COLS
    detector.grid_rows = GRID_ROWS
    
    # Save detection steps if requested
    if SAVE_DETECTION_STEPS:
        print("Generating detection steps visualization...")
        detection_data = detector.get_detection_data(first_frame, BRIGHTNESS_THRESHOLD)
        steps_output = output_dir / "detection_steps.png"
        VisualizationUtils.save_detection_steps_visualization(
            detection_data, first_frame, str(steps_output)
        )
        print()
    
    # Create visualizer instance
    visualizer = VisualizationUtils() if VISUALIZE else None
    
    # Process video
    print("Starting video processing...")
    results = detector.process_video(
        output_path=output_video if VISUALIZE else None,
        visualize=VISUALIZE,
        show_cell_labels=SHOW_LABELS,
        brightness_threshold=BRIGHTNESS_THRESHOLD,
        visualizer=visualizer
    )
    
    if results is None:
        print("\nError: Detection failed")
        return 1
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    
    # List output files
    print("\nGenerated files:")
    if VISUALIZE:
        print(f"  - Output video: {output_video}")
    
    results_json = video_path.parent / f"{video_path.stem}_board_detection.json"
    print(f"  - Detection results: {results_json}")
    
    if SAVE_DETECTION_STEPS:
        print(f"  - Detection steps: {output_dir / 'detection_steps.png'}")
    
    if SAVE_FILTER_ANALYSIS:
        print(f"  - Filter analysis: {output_dir / 'filter_analysis.png'}")
    
    print("\n✓ All done!")
    return 0


if __name__ == "__main__":
    exit(main())