import cv2
import numpy as np
from scipy import ndimage
from collections import deque


class DiceDetector:
    """Detect dice on the game board and track them across frames."""
    
    def __init__(self, history_length=20, distance_threshold=30, dice_radius=40):
        """
        Initialize the dice detector.
        
        Args:
            history_length: Number of frames to track dice for stability
            distance_threshold: Max pixel distance for matching dice across frames
            dice_radius: Radius for counting dots on dice faces
        """
        self.history_length = history_length
        self.distance_threshold = distance_threshold
        self.dice_radius = dice_radius
        
        # Tracking history
        self.history = deque(maxlen=history_length)
        self.score_history = deque(maxlen=history_length)
        
    def _create_component_size_descriptor(self, binary_mask):
        """Create a descriptor where each pixel contains the size of its connected component."""
        binary_mask = binary_mask.astype(bool)
        labeled_array, num_features = ndimage.label(binary_mask)
        component_sizes = {}
        for label in range(1, num_features + 1):
            size = np.sum(labeled_array == label)
            component_sizes[label] = size
        
        descriptor = np.zeros_like(binary_mask, dtype=np.int32)
        for label, size in component_sizes.items():
            descriptor[labeled_array == label] = size
        
        return descriptor
    
    def _shrink_to_single_pixels(self, image):
        """Reduce each connected component to its centroid."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        result = np.zeros_like(image)
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            cx, cy = int(round(cx)), int(round(cy))
            if 0 <= cy < result.shape[0] and 0 <= cx < result.shape[1]:
                result[cy, cx] = 255
        return result
    
    def _extend_corners_horizontally(self, corners):
        """Extend corners horizontally by 15% of the width on each side."""
        extended = corners.copy()
        
        # Bottom edge
        bottom_left = corners[3]
        bottom_right = corners[2]
        bottom_diff = bottom_right[0] - bottom_left[0]
        extended[3, 0] = bottom_left[0] - bottom_diff * 0.15
        extended[2, 0] = bottom_right[0] + bottom_diff * 0.15
        
        # Top edge
        top_left = corners[0]
        top_right = corners[1]
        top_diff = top_right[0] - top_left[0]
        extended[0, 0] = top_left[0] - top_diff * 0.15
        extended[1, 0] = top_right[0] + top_diff * 0.15
        
        return extended
    
    def _count_black_regions_in_circle(self, binary_image, center_x, center_y, radius):
        """
        Count the number of distinct black regions within a circular area.
        
        Args:
            binary_image: Binary image (255 = white, 0 = black)
            center_x, center_y: Center coordinates of the circle
            radius: Radius of the circular area to examine
        
        Returns:
            Number of distinct black regions (connected components)
        """
        h, w = binary_image.shape
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Extract region within circle
        circular_region = np.zeros_like(binary_image)
        circular_region[mask] = binary_image[mask]
        
        # Invert to make black regions white (for connected components)
        inverted_region = cv2.bitwise_not(circular_region)
        inverted_region[~mask] = 0
        
        # Find connected components in black regions
        num_labels = cv2.connectedComponentsWithStats(inverted_region, connectivity=8)[0]
        
        # Subtract 1 to exclude background
        return max(0, num_labels - 1)
    
    def _detect_dice_centers(self, frame, board_corners):
        """
        Detect dice centers in a single frame.
        
        Args:
            frame: Input frame (BGR)
            board_corners: 4x2 numpy array of board corner coordinates
                          Order: [top_left, top_right, bottom_right, bottom_left]
        
        Returns:
            kostki_centers: Binary image with center pixels at 255
            dot_mask: Binary mask for dot counting
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        th_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        th_adjusted = th_otsu * 0.8
        _, mask1 = cv2.threshold(blurred, th_adjusted, 255, cv2.THRESH_BINARY)
        
        # Morphological closing
        kernel = np.ones((7, 7), np.uint8)
        mask1_closed = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel, iterations=4)
        
        # Extend board corners and create board mask
        extended_corners = self._extend_corners_horizontally(board_corners)
        mask_board = np.zeros(frame.shape[:2], dtype=np.uint8)
        h, w = frame.shape[:2]
        
        def get_x_at_y(corner_bottom, corner_top, y):
            if corner_top[1] == corner_bottom[1]:
                return corner_bottom[0]
            slope = (corner_top[0] - corner_bottom[0]) / (corner_top[1] - corner_bottom[1])
            return corner_bottom[0] + slope * (y - corner_bottom[1])
        
        x_left_top = get_x_at_y(extended_corners[3], extended_corners[0], 0)
        x_left_bottom = get_x_at_y(extended_corners[3], extended_corners[0], h)
        x_right_top = get_x_at_y(extended_corners[2], extended_corners[1], 0)
        x_right_bottom = get_x_at_y(extended_corners[2], extended_corners[1], h)
        
        quad = np.array([
            [x_left_top, 0],
            [x_right_top, 0],
            [x_right_bottom, h],
            [x_left_bottom, h]
        ], dtype=np.int32)
        
        cv2.fillPoly(mask_board, [quad], 255)
        mask1_closed = mask1_closed | mask_board
        
        # Second threshold for dice detection
        th_adjusted = th_otsu * 1.4
        _, mask2 = cv2.threshold(blurred, th_adjusted, 255, cv2.THRESH_BINARY)
        mask2_closed = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Filter by component size
        component_size_desc = self._create_component_size_descriptor(mask1_closed)
        
        max_threshold = 3000
        min_threshold = 400
        kostki = mask2_closed & (component_size_desc <= max_threshold) & (component_size_desc > min_threshold)
        kostki_centers = self._shrink_to_single_pixels(kostki)
        
        # Create binary mask for dot counting
        _, dot_mask = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
        
        return kostki_centers, dot_mask
    
    def detect_dice(self, frame, board_corners):
        """
        Detect dice in the current frame and return stable dice with their values.
        
        Args:
            frame: Input frame (BGR)
            board_corners: 4x2 numpy array of board corner coordinates
        
        Returns:
            List of dicts with keys: 'x', 'y', 'value' for each stable dice
        """
        # Detect dice centers in current frame
        kostki_centers, binary_mask = self._detect_dice_centers(frame, board_corners)
        
        # Get all detected centers
        y_coords, x_coords = np.where(kostki_centers == 255)
        
        # Compute scores for all detected centers
        scores_dict = {}
        for x, y in zip(x_coords, y_coords):
            num_dots = self._count_black_regions_in_circle(
                binary_mask, x, y, self.dice_radius
            )
            # Subtract 1 to exclude the area around die
            score = max(0, num_dots - 1)
            scores_dict[(x, y)] = score
        
        # Create list of current centers
        current_centers = list(zip(x_coords, y_coords))
        current_scores = [(x, y, scores_dict.get((x, y), 0)) for x, y in current_centers]
        
        # Add to history
        self.history.append(current_centers)
        self.score_history.append(current_scores)
        
        # Get stable centers if we have enough history
        stable_dice = self._get_stable_centers_with_scores()
        
        return stable_dice
    
    def _get_stable_centers_with_scores(self):
        """
        Get centers that are stable across the history window with averaged scores.
        
        Returns:
            List of dicts with 'x', 'y', 'value' for stable dice
        """
        if len(self.history) < self.history_length:
            return []
        
        # Get current frame centers
        current_centers = self.history[-1]
        stable_dice = []
        
        for current_x, current_y in current_centers:
            # Try to track this center across all frames in history
            matched_positions = []
            matched_scores = []
            
            for frame_idx in range(len(self.history)):
                frame_centers = self.history[frame_idx]
                
                if not frame_centers:
                    break
                
                # Find closest center in this historical frame
                distances = [
                    np.sqrt((current_x - x)**2 + (current_y - y)**2)
                    for x, y in frame_centers
                ]
                min_dist = min(distances)
                min_idx = distances.index(min_dist)
                
                # If closest center is within threshold, consider it a match
                if min_dist <= self.distance_threshold:
                    matched_positions.append(frame_centers[min_idx])
                    
                    # Get the score for this matched position
                    matched_x, matched_y = frame_centers[min_idx]
                    for x, y, score in self.score_history[frame_idx]:
                        if x == matched_x and y == matched_y:
                            matched_scores.append(score)
                            break
                else:
                    break
            
            # If we found matches in all frames, calculate averages
            if len(matched_positions) == self.history_length:
                avg_x = np.mean([x for x, y in matched_positions])
                avg_y = np.mean([y for x, y in matched_positions])
                
                # Check if current position is close to average
                dist_from_avg = np.sqrt((current_x - avg_x)**2 + (current_y - avg_y)**2)
                
                if dist_from_avg <= self.distance_threshold:
                    # Calculate average score and round it
                    avg_score = int(round(np.mean(matched_scores)))
                    
                    stable_dice.append({
                        'x': int(avg_x),
                        'y': int(avg_y),
                        'value': avg_score
                    })
        
        return stable_dice
    
    def reset(self):
        """Reset the tracker history."""
        self.history.clear()
        self.score_history.clear()