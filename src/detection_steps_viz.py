import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

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

def save_token_detection_visualization(output_dir: Path, frame_number: int,
                                      frame: np.ndarray, hsv: np.ndarray,
                                      mask1: np.ndarray, mask2: np.ndarray,
                                      red_mask_raw: np.ndarray,
                                      red_mask_open: np.ndarray,
                                      red_mask_close: np.ndarray,
                                      red_mask_dilate: np.ndarray,
                                      red_mask_before_blur: np.ndarray,
                                      red_mask_final: np.ndarray,
                                      search_regions: List[Tuple],
                                      detected_circles: List[Tuple],
                                      battles: List,
                                      min_radius: int, max_radius: int, min_dist: int):
    """
    Save step-by-step visualization of token detection process (12 steps)
    """
    
    # Convert HSV back to BGR for visualization
    hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Create search regions visualization
    search_region_vis = frame.copy()
    for region, battle_id in search_regions:
        sx, sy, sw, sh = region
        cv2.rectangle(search_region_vis, (sx, sy), (sx + sw, sy + sh), (255, 0, 255), 2)
        cv2.putText(search_region_vis, f"B{battle_id}", (sx + 5, sy + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Draw battles
    for battle in battles:
        cv2.drawContours(search_region_vis, [battle.box], 0, (0, 0, 255), 2)
    
    # Create final result with detected circles and team assignment
    final_result = frame.copy()
    for cx, cy, radius, team, battle_id in detected_circles:
        color = (255, 0, 255) if team == 'A' else (255, 255, 0)  # Violet for A, Cyan for B
        cv2.circle(final_result, (cx, cy), radius, color, 3)
        cv2.circle(final_result, (cx, cy), 3, color, -1)
        label = f"T{team}"
        cv2.putText(final_result, label, (cx + radius + 5, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Create 3x4 grid (12 images)
    row1 = np.hstack([
        frame.copy(),
        hsv_vis,
        cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
    ])
    
    row2 = np.hstack([
        cv2.cvtColor(red_mask_raw, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(red_mask_open, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(red_mask_close, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(red_mask_dilate, cv2.COLOR_GRAY2BGR)
    ])
    
    row3 = np.hstack([
        cv2.cvtColor(red_mask_before_blur, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(red_mask_final, cv2.COLOR_GRAY2BGR),
        search_region_vis,
        final_result
    ])
    
    grid = np.vstack([row1, row2, row3])
    
    # Add labels
    labels = [
        ["1. Original", "2. HSV", "3. Red Mask 1 (0-10°)", "4. Red Mask 2 (160-180°)"],
        ["5. Combined Red Mask", "6. After Open (5x5)", "7. After Close (7x7)", "8. After Dilate (5x5)"],
        ["9. Before Blur", "10. After Blur (3x3)", "11. Search Regions", "12. Hough Circles + Teams"]
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    for i, row_labels in enumerate(labels):
        for j, label in enumerate(row_labels):
            y = i * h + 25
            x = j * w + 10
            cv2.putText(grid, label, (x, y), font, 0.5, (0, 0, 0), 3)
            cv2.putText(grid, label, (x, y), font, 0.5, (0, 255, 255), 1)
    
    # Add summary panel
    summary_height = 100
    summary_panel = np.zeros((summary_height, grid.shape[1], 3), dtype=np.uint8)
    
    cv2.putText(summary_panel, "Token Detection Pipeline - 12 Steps", 
               (20, 25), font, 0.7, (0, 255, 255), 2)
    cv2.putText(summary_panel, f"Frame {frame_number} | Detected: {len(detected_circles)} tokens | Battles: {len(battles)}", 
               (20, 50), font, 0.6, (255, 255, 255), 1)
    cv2.putText(summary_panel, f"Params: minR={min_radius}, maxR={max_radius}, minDist={min_dist}", 
               (20, 75), font, 0.5, (200, 200, 200), 1)
    
    final_grid = np.vstack([grid, summary_panel])
    
    # Save to file
    output_path = output_dir / f"token_frame_{frame_number:06d}.jpg"
    cv2.imwrite(str(output_path), final_grid)
    return output_path