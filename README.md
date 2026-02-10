# Summoner Wars Gameplay Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24.3-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.11.2-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![uv](https://img.shields.io/badge/uv-Package_Manager-00ADD8?style=for-the-badge)

**Computer vision system for detecting board game elements, tracking movements, and recognizing gameplay events from video**

[Dataset & Results](https://drive.google.com/drive/folders/1lridoDl2Of4L5lEuyWcF35dPPpg4n7ap) • [Documentation](project-overview.pdf)

</div>

---

## 🎯 Overview

A comprehensive computer vision pipeline for real-time detection and tracking of Summoner Wars gameplay elements. The system identifies game boards, playing cards, battle tokens, and dice from video footage, automatically tracking game state and recognizing key events.

### Key Features

- **Multi-object Detection**: Tracks boards, cards (Team A/B), battle tokens, and dice simultaneously
- **Adaptive Board Detection**: Handles camera angles, movement, and hand occlusions with robust boundary tracking
- **Battle Recognition**: Automatically detects card battles based on proximity and team assignment
- **Token Tracking**: Identifies hit indicators with 15-frame confirmation to prevent false positives
- **Dice Reading**: Detects dice values (1-6) with temporal smoothing across 20 frames
- **Event System**: Monitors 7+ game events including setup, movements, battles, and win conditions

---

## 🎮 Game Elements Detected

| Element | Detection Method | Key Features |
|---------|-----------------|--------------|
| **Board** | Otsu thresholding + morphological ops | Perspective correction, occlusion handling |
| **Cards** | Watershed segmentation + KCF tracking | Team assignment, IoU matching (0.3 threshold) |
| **Battles** | Geometric proximity detection | 10% buffer zone, dynamic bounding boxes |
| **Tokens** | HSV color segmentation + Hough circles | Battle-specific search regions, 15-frame confirmation |
| **Dice** | Adaptive thresholding + component analysis | Temporal tracking (20 frames), dot counting |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

Clone the repository:
```
git clone <repository-url>
cd gameplay-event-detection
```

Install dependencies with uv:
```
uv sync
```

### Dataset Setup

1. Download data from [Google Drive](https://drive.google.com/drive/folders/1lridoDl2Of4L5lEuyWcF35dPPpg4n7ap)
2. Extract to `data/` directory:
```
data/
├── easy/
├── medium/
└── hard/
```

### Running Detection

Process a video:
```
python src/run.py
```

Configure video path in run.py:
```
VIDEO_PATH = "data/medium/medium3.mp4"  # Change as needed
```

**Output**: Annotated video saved to `output_video_tokens/detected_<filename>.mp4`

---

## 🔧 Detection Pipeline

### 1. Board Detection
- **Preprocessing**: Gaussian blur → Otsu threshold (×110%) → morphological opening
- **Edge Refinement**: Scans inward from detected rectangle edges to find 80% white pixel density
- **Perspective Correction**: Adjusts corners to handle angled views
- **Occlusion Handling**: Rejects detections when hand interference detected (threshold ×1.1 on repeated failures)

### 2. Card Detection & Tracking

Detection: Canny edges → dilation → watershed segmentation

Filtering: Area (10k-15k px), aspect ratio (1.3-2.4), ORB keypoints (≥15)

Tracking: KCF trackers with IoU matching (threshold 0.3)

Team assignment: Based on board center position

### 3. Battle Detection
- Triggered when opposing team cards' bounding boxes (expanded by 10%) intersect
- Maintains battle center and dynamic bounding box
- Resolves winner by detecting which card leaves battle zone first

### 4. Token Detection

HSV red segmentation (150-180° hue, 30-255 saturation/value)

Morphological: Opening (5×5) → Closing (7×7) → Dilation (5×5)

Hough circles: Radius 0.9-2.0% of board width, search above/below battles

Confirmation: 15 consecutive frames before counting as hit

### 5. Dice Detection
- **Dual thresholding**: Otsu ×0.8 (mask1) and ×1.4 (mask2) with morphological closing
- **Component filtering**: Size 400-3000 pixels using custom descriptor
- **Value reading**: Counts black regions in circular area (radius=40px)
- **Temporal smoothing**: 20-frame history with distance threshold=30px

---

## 📊 Game Events Tracked

1. **Setup Complete**: Triggered after 4 card appearances (2 per team)
2. **Card Movement**: Detects displacement >30 pixels
3. **Battle Start**: New opposing card proximity
4. **Token Hits**: Confirmed hits with team attribution
5. **Battle End**: Card removal with winner determination
6. **Score Updates**: Real-time team health tracking
7. **Game Over**: Win condition when one team has 0 cards

---

## 📁 Project Structure
```
gameplay-event-detection/
├── src/
│   ├── board_detector.py        # Board boundary detection
│   ├── card_detector.py         # Card detection + tracking
│   ├── token_detector.py        # Battle token detection
│   ├── dice_detector.py         # Dice detection + value reading
│   ├── visualization.py         # Rendering functions
│   └── run.py                   # Main pipeline
├── data/                        # Video dataset (download separately)
├── output_video_tokens/         # Processed output videos
├── output_visualization_*/      # Debug visualizations (every 200 frames)
├── pyproject.toml              # uv dependencies
└── README.md
```

---

## 🔬 Technical Details

### Dependencies
- **OpenCV**: Morphological ops, Hough transforms, KCF tracking
- **NumPy/SciPy**: Connected components, distance transforms
- **scikit-image**: Watershed segmentation
- **Pillow**: Image I/O

### Key Parameters

Dice Detection:
```
HISTORY_LENGTH = 20        # Temporal smoothing window
DISTANCE_THRESHOLD = 30    # Max pixel movement between frames
DICE_RADIUS = 40          # Region for dot counting
```

Token Detection:
```
CONFIRMATION_FRAMES = 15   # Stability before counting hit
MIN_DISTANCE = 2% board width  # Prevents duplicate circles
```

Card Tracking:
```
IOU_THRESHOLD = 0.3       # Match existing tracks
MAX_LOST_FRAMES = 15      # Before removing track
```

---

## 🎯 Known Limitations

1. **Dice Value Confusion**: 5 vs 6 sometimes misclassified (acceptable as both result in token award ≥3)
2. **Hand Occlusions**: Brief tracking loss during card placement (mitigated by 15-frame grace period)
3. **Camera Shake**: Reduced accuracy on "hard" dataset with irregular movements
4. **Board Loss**: Rare full board detection failures when outer board elements occlude grid

---

## 📄 License

See project documentation for details.

---

## 🔗 Resources

- **Dataset & Output Videos**: [Google Drive](https://drive.google.com/drive/folders/1lridoDl2Of4L5lEuyWcF35dPPpg4n7ap)
- **Detailed Report**: [project-overview.pdf](project-overview.pdf)
- **uv Documentation**: [astral.sh/uv](https://docs.astral.sh/uv/)
