import cv2
from pathlib import Path
from typing import Tuple, Optional, List


class VideoLoader:
    """Handles video file loading and frame extraction"""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        
    def open(self) -> bool:
        """Open video file and read properties"""
        if not self.video_path.exists():
            print(f"❌ Video not found: {self.video_path}")
            return False
            
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            return False
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Loaded: {self.video_path.name} ({self.width}x{self.height}, {self.fps}fps, {self.frame_count} frames)")
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[any]]:
        """Read next frame"""
        return self.cap.read() if self.cap else (False, None)
    
    def get_frame_number(self) -> int:
        """Get current frame position"""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.cap else 0
    
    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()


class DatasetLoader:
    """Manages loading videos from dataset directory"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.videos = {'easy': [], 'medium': [], 'difficult': []}
        
    def scan(self) -> bool:
        """Scan for video files"""
        if not self.data_dir.exists():
            print(f"❌ Directory not found: {self.data_dir}")
            self._create_structure()
            return False
        
        print(f"\n📁 Scanning: {self.data_dir}")
        extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        for difficulty in self.videos.keys():
            diff_dir = self.data_dir / difficulty
            if diff_dir.exists():
                for ext in extensions:
                    self.videos[difficulty].extend(diff_dir.glob(f'*{ext}'))
                print(f"  [{difficulty}] {len(self.videos[difficulty])} video(s)")
        
        total = sum(len(v) for v in self.videos.values())
        print(f"  Total: {total} video(s)\n")
        return total > 0
    
    def _create_structure(self):
        """Create directory structure"""
        for difficulty in self.videos.keys():
            (self.data_dir / difficulty).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {self.data_dir}/{{easy,medium,difficult}}/")
    
    def get_all(self) -> List[Tuple[str, Path]]:
        """Get all videos with their difficulty"""
        return [(d, v) for d, videos in self.videos.items() for v in videos]