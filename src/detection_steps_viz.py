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
    dist_vis = cv2.applyColorMap(dist_vis, cv2.COLORMAP_JET)
    
    # Create markers visualization
    markers_color = np.zeros_like(frame)
    num_cards_detected = 0
    for label in np.unique(markers_post):
        if label == -1:
            markers_color[markers_post == -1] = [0, 0, 255] 
        elif label > 1:
            num_cards_detected += 1
            color = np.random.randint(0, 255, 3).tolist()
            markers_color[markers_post == label] = color
    
    # Create final detection overlay
    final_detection = frame.copy()
    for card in detected_cards:
        cv2.drawContours(final_detection, [card.box], -1, (0, 255, 0), 3)
        center = (int(card.center[0]), int(card.center[1]))
        cv2.circle(final_detection, center, 5, (0, 0, 255), -1)
    
    # Prepare 6 images
    images = [
        (frame.copy(), "Original Frame"),
        (cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "Edge Detection (Canny)"),
        (cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), "Filled Mask"),
        (dist_vis, "Distance Transform"),
        (cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR), "Foreground Markers"),
        (markers_color, "Watershed Segmentation"),
    ]
    
    # Add titles to each image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    for img, title in images:
        # Add black background for text
        cv2.rectangle(img, (5, 5), (400, 40), (0, 0, 0), -1)
        cv2.putText(img, title, (10, 30), font, font_scale, (0, 255, 255), thickness)
    
    # Create 3x2 grid
    row1 = np.hstack([images[0][0], images[1][0]])
    row2 = np.hstack([images[2][0], images[3][0]])
    row3 = np.hstack([images[4][0], images[5][0]])
    
    grid = np.vstack([row1, row2, row3])
    
    # Add final detection as bottom panel
    final_panel = final_detection.copy()
    cv2.rectangle(final_panel, (5, 5), (400, 40), (0, 0, 0), -1)
    cv2.putText(final_panel, "Final Detection", (10, 30), font, font_scale, (0, 255, 255), thickness)
    
    # Resize final_panel to match grid width (duplicate horizontally)
    final_panel_wide = np.hstack([final_panel, final_panel])
    
    # Stack vertically
    final_grid = np.vstack([grid, final_panel_wide])
    
    # Add summary at bottom
    summary_height = 60
    summary_panel = np.zeros((summary_height, final_grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(summary_panel, f"Frame {frame_number} | Cards Detected: {len(detected_cards)} | Watershed Regions: {num_cards_detected}", 
               (20, 35), font, 0.6, (255, 255, 255), 1)
    
    final_grid = np.vstack([final_grid, summary_panel])
    
    # Save to file
    output_path = output_dir / f"frame_{frame_number:06d}.jpg"
    cv2.imwrite(str(output_path), final_grid)
    return output_path


def save_token_detection_visualization(output_dir: Path, frame_number: int,
                                      frame: np.ndarray, hsv: np.ndarray,
                                      red_mask: np.ndarray,
                                      red_mask_open: np.ndarray,
                                      red_mask_close: np.ndarray,
                                      red_mask_final: np.ndarray,
                                      search_regions: List[Tuple],
                                      detected_circles: List[Tuple],
                                      battles: List,
                                      min_radius: int, max_radius: int, min_dist: int):
    # Convert HSV back to BGR for visualization
    hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Create search regions visualization
    search_region_vis = frame.copy()
    for region, battle_id in search_regions:
        sx, sy, sw, sh = region
        cv2.rectangle(search_region_vis, (sx, sy), (sx + sw, sy + sh), (255, 0, 255), 2)
        cv2.putText(search_region_vis, f"Battle {battle_id}", (sx + 5, sy + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Draw battle boxes
    for battle in battles:
        cv2.drawContours(search_region_vis, [battle.box], 0, (0, 255, 0), 2)
    
    # Create final result with detected circles and team assignment
    final_result = frame.copy()
    for cx, cy, radius, team, battle_id in detected_circles:
        color = (255, 0, 255) if team == 'A' else (255, 255, 0)  # Violet for A, Cyan for B
        cv2.circle(final_result, (cx, cy), radius, color, 3)
        cv2.circle(final_result, (cx, cy), 3, color, -1)
        label = f"Team {team}"
        cv2.putText(final_result, label, (cx + radius + 5, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Prepare 6 images
    images = [
        (frame.copy(), "Original Frame"),
        (hsv_vis, "HSV Color Space"),
        (cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR), "Red Color Mask (150-180)"),
        (cv2.cvtColor(red_mask_close, cv2.COLOR_GRAY2BGR), "After Morphology (Open+Close)"),
        (search_region_vis, "Battle Search Regions"),
        (final_result, "Detected Tokens + Teams"),
    ]
    
    # Add titles to each image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    for img, title in images:
        # Add black background for text
        cv2.rectangle(img, (5, 5), (500, 40), (0, 0, 0), -1)
        cv2.putText(img, title, (10, 30), font, font_scale, (0, 255, 255), thickness)
    
    # Create 3x2 grid
    row1 = np.hstack([images[0][0], images[1][0]])
    row2 = np.hstack([images[2][0], images[3][0]])
    row3 = np.hstack([images[4][0], images[5][0]])
    
    grid = np.vstack([row1, row2, row3])
    
    # Add summary at bottom - WIDTH MATCHES GRID
    summary_height = 80
    summary_panel = np.zeros((summary_height, grid.shape[1], 3), dtype=np.uint8)
    
    cv2.putText(summary_panel, f"Frame {frame_number} | Tokens Detected: {len(detected_circles)} | Active Battles: {len(battles)}", 
               (20, 30), font, 0.6, (255, 255, 255), 1)
    cv2.putText(summary_panel, f"Parameters: Min Radius={min_radius}px, Max Radius={max_radius}px, Min Distance={min_dist}px", 
               (20, 55), font, 0.5, (200, 200, 200), 1)
    
    final_grid = np.vstack([grid, summary_panel])
    
    # Save to file
    output_path = output_dir / f"token_frame_{frame_number:06d}.jpg"
    cv2.imwrite(str(output_path), final_grid)
    return output_path