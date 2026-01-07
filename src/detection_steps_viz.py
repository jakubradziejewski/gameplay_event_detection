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


def save_board_detection_visualization(output_dir: Path, frame_number: int,
                                      frame: np.ndarray,
                                      mask_otsu: np.ndarray,
                                      mask_opened: np.ndarray,
                                      largest_contour,
                                      bounding_rect_corners: np.ndarray,
                                      corners_after_white_threshold: np.ndarray,
                                      final_corners: np.ndarray):
    
    # Image 1: Binary mask after Otsu
    otsu_vis = cv2.cvtColor(mask_otsu, cv2.COLOR_GRAY2BGR)
    
    # Image 2: After morphological opening
    opened_vis = cv2.cvtColor(mask_opened, cv2.COLOR_GRAY2BGR)
    
    # Image 3: Largest contour on original image
    contour_vis = frame.copy()
    if largest_contour is not None:
        cv2.drawContours(contour_vis, [largest_contour], -1, (0, 255, 0), 3)
        # Calculate and display contour area
        area = cv2.contourArea(largest_contour)
        frame_area = frame.shape[0] * frame.shape[1]
        area_percent = area / frame_area * 100
        cv2.putText(contour_vis, f"Area: {area_percent:.1f}%", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Image 4: Bounding rectangle
    bounding_vis = frame.copy()
    if bounding_rect_corners is not None:
        corners_int = bounding_rect_corners.astype(np.int32)
        cv2.polylines(bounding_vis, [corners_int], True, (255, 0, 0), 3)
        # Draw corner points
        for i, corner in enumerate(corners_int):
            cv2.circle(bounding_vis, tuple(corner), 8, (255, 0, 0), -1)
            cv2.putText(bounding_vis, str(i), (corner[0] + 10, corner[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Image 5: After white threshold edge detection
    white_thresh_vis = frame.copy()
    if corners_after_white_threshold is not None:
        corners_int = corners_after_white_threshold.astype(np.int32)
        cv2.polylines(white_thresh_vis, [corners_int], True, (0, 255, 255), 3)
        # Draw corner points
        for i, corner in enumerate(corners_int):
            cv2.circle(white_thresh_vis, tuple(corner), 8, (0, 255, 255), -1)
            cv2.putText(white_thresh_vis, str(i), (corner[0] + 10, corner[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Image 6: Final detection after edge refinement
    final_vis = frame.copy()
    if final_corners is not None:
        corners_int = final_corners.astype(np.int32)
        cv2.polylines(final_vis, [corners_int], True, (0, 255, 0), 3)
        # Draw corner points
        for i, corner in enumerate(corners_int):
            cv2.circle(final_vis, tuple(corner), 8, (0, 255, 0), -1)
            cv2.putText(final_vis, str(i), (corner[0] + 10, corner[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Prepare 6 images with titles
    images = [
        (otsu_vis, "1. Binary Mask (Otsu)"),
        (opened_vis, "2. After Morphological Opening"),
        (contour_vis, "3. Largest Contour"),
        (bounding_vis, "4. Bounding Rectangle"),
        (white_thresh_vis, "5. After White Threshold"),
        (final_vis, "6. Final Detection (Refined)"),
    ]
    
    # Add titles to each image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    for img, title in images:
        # Add black background for text
        cv2.rectangle(img, (5, 5), (450, 40), (0, 0, 0), -1)
        cv2.putText(img, title, (10, 30), font, font_scale, (0, 255, 255), thickness)
    
    # Create 2x3 grid
    row1 = np.hstack([images[0][0], images[1][0], images[2][0]])
    row2 = np.hstack([images[3][0], images[4][0], images[5][0]])
    
    grid = np.vstack([row1, row2])
    
    # Add summary at bottom
    summary_height = 60
    summary_panel = np.zeros((summary_height, grid.shape[1], 3), dtype=np.uint8)
    
    detection_status = "SUCCESS" if final_corners is not None else "FAILED"
    status_color = (0, 255, 0) if final_corners is not None else (0, 0, 255)
    
    cv2.putText(summary_panel, f"Frame {frame_number} | Board Detection: {detection_status}", 
               (20, 35), font, 0.6, status_color, 2)
    
    final_grid = np.vstack([grid, summary_panel])
    
    # Save to file
    output_path = output_dir / f"board_frame_{frame_number:06d}.jpg"
    cv2.imwrite(str(output_path), final_grid)
    return output_path


def save_dice_detection_visualization(output_dir: Path, frame_number: int,
                                     frame: np.ndarray,
                                     board_corners: np.ndarray,
                                     mask_board: np.ndarray,
                                     mask1: np.ndarray,
                                     mask1_closed_before: np.ndarray,
                                     mask1_closed_after: np.ndarray,
                                     mask2: np.ndarray,
                                     mask2_closed: np.ndarray,
                                     dice_filtered: np.ndarray,
                                     dice_centers: np.ndarray,
                                     detected_dice: list):
    
    # Image 1: Original with board corners
    board_vis = frame.copy()
    corners_int = board_corners.astype(np.int32)
    cv2.polylines(board_vis, [corners_int], True, (0, 255, 0), 2)
    for i, corner in enumerate(corners_int):
        cv2.circle(board_vis, tuple(corner), 5, (0, 255, 0), -1)
    
    # Image 2: Mask board (extended corners filled)
    mask_board_vis = cv2.cvtColor(mask_board, cv2.COLOR_GRAY2BGR)
    
    # Image 3: First mask
    mask1_vis = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
    
    # Image 4: First mask closed (before merge)
    mask1_closed_before_vis = cv2.cvtColor(mask1_closed_before, cv2.COLOR_GRAY2BGR)
    
    # Image 5: Mask 1 after merging with board mask
    mask1_closed_after_vis = cv2.cvtColor(mask1_closed_after, cv2.COLOR_GRAY2BGR)
    
    # Image 6: Mask 2
    mask2_vis = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
    
    # Image 7: Mask 2 closed
    mask2_closed_vis = cv2.cvtColor(mask2_closed, cv2.COLOR_GRAY2BGR)
    
    # Image 8: Dice filtered (after threshold)
    dice_filtered_vis = cv2.cvtColor(dice_filtered, cv2.COLOR_GRAY2BGR)
    
    # Image 9: Dice centers on original
    dice_centers_vis = frame.copy()
    for dice in detected_dice:
        x, y, value = dice['x'], dice['y'], dice['value']
        # Circle on the die
        cv2.circle(dice_centers_vis, (x, y), 18, (0, 255, 0), 2)
        # Value label directly above each die
        cv2.putText(dice_centers_vis, str(value), (x - 10, y - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Add info block with dice information
    if detected_dice:
        start_y = 120
        # Semi-transparent background for the text block
        overlay = dice_centers_vis.copy()
        cv2.rectangle(overlay, (10, start_y), (350, start_y + 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, dice_centers_vis, 0.5, 0, dice_centers_vis)
        
        # Dice Count
        cv2.putText(dice_centers_vis, f'Total Dice: {len(detected_dice)}', (20, start_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Individual Scores List
        scores = [str(die['value']) for die in detected_dice]
        score_str = f"Scores: {', '.join(scores)}"
        cv2.putText(dice_centers_vis, score_str, (20, start_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Prepare 9 images with titles
    images = [
        (board_vis, "1. Original + Board Corners"),
        (mask_board_vis, "2. Board Mask (Extended)"),
        (mask1_vis, "3. Mask 1 (Otsu * 0.8)"),
        (mask1_closed_before_vis, "4. Mask 1 Closed (Before Merge)"),
        (mask1_closed_after_vis, "5. Mask 1 After Board Merge"),
        (mask2_vis, "6. Mask 2 (Otsu * 1.4)"),
        (mask2_closed_vis, "7. Mask 2 Closed"),
        (dice_filtered_vis, "8. Dice Filtered (Size Threshold)"),
        (dice_centers_vis, "9. Final Dice Centers + Values"),
    ]
    
    # Add titles to each image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    for img, title in images:
        # Add black background for text
        cv2.rectangle(img, (5, 5), (450, 35), (0, 0, 0), -1)
        cv2.putText(img, title, (10, 25), font, font_scale, (0, 255, 255), thickness)
    
    # Create 3x3 grid
    row1 = np.hstack([images[0][0], images[1][0], images[2][0]])
    row2 = np.hstack([images[3][0], images[4][0], images[5][0]])
    row3 = np.hstack([images[6][0], images[7][0], images[8][0]])
    
    grid = np.vstack([row1, row2, row3])
    
    # Add summary at bottom
    summary_height = 60
    summary_panel = np.zeros((summary_height, grid.shape[1], 3), dtype=np.uint8)
    
    cv2.putText(summary_panel, f"Frame {frame_number} | Dice Detected: {len(detected_dice)}", 
               (20, 25), font, 0.6, (255, 255, 255), 2)
    
    if detected_dice:
        values_str = "Values: " + ", ".join([str(d['value']) for d in detected_dice])
        cv2.putText(summary_panel, values_str, (20, 50), font, 0.5, (200, 200, 200), 1)
    
    final_grid = np.vstack([grid, summary_panel])
    
    # Save to file
    output_path = output_dir / f"dice_frame_{frame_number:06d}.jpg"
    cv2.imwrite(str(output_path), final_grid)
    return output_path