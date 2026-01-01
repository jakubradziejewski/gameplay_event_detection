import cv2
import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from data_loader import VideoLoader, DatasetLoader


def main():
    print("=" * 60)
    print("BOARD GAME DETECTION SYSTEM")
    print("=" * 60)

    dataset = DatasetLoader(data_dir="data")

    # Scan for videos
    if not dataset.scan():
        print("\n⚠ No videos found in dataset!")
        print("Please add video files to the data directory.")
        return

    # Grab all videos (difficulty, path)
    all_videos = dataset.get_all()

    if not all_videos:
        print("❌ No videos available to process")
        return

    difficulty, video_path = all_videos[0]
    print(f"\nSelected video: {video_path.name} (Difficulty: {difficulty})\n")

    # Open video
    video = VideoLoader(str(video_path))

    if not video.open():
        print("❌ Failed to open video")
        return

    # Derived info
    duration = video.frame_count / video.fps if video.fps > 0 else 0.0

    print("=" * 60)
    print("VIDEO PLAYBACK")
    print("=" * 60)
    print("Controls:")
    print("  q - quit")
    print("  space - pause/resume")
    print("  r - restart")
    print("=" * 60)

    cv2.namedWindow("Video Playback", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Playback", 1280, 720)

    paused = False
    frames_displayed = 0

    try:
        while True:
            if not paused:
                ret, frame = video.read_frame()
                if not ret:
                    print("\n✓ End of video reached")
                    break

                frames_displayed += 1

                current = video.get_frame_number()
                progress = (current / video.frame_count) * 100 if video.frame_count else 0

                info = f"Frame: {current}/{video.frame_count} ({progress:.1f}%)"
                cv2.putText(frame, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                status = "PAUSED" if paused else "PLAYING"
                cv2.putText(frame, status, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Video Playback", frame)

                delay = max(1, int(1000 / video.fps)) if video.fps > 0 else 30
            else:
                delay = 100

            key = cv2.waitKey(delay) & 0xFF

            if key == ord("q"):
                print("\n⏹ Quit by user")
                break

            elif key == ord(" "):
                paused = not paused
                print("⏸ Paused" if paused else "▶ Resumed")

            elif key == ord("r"):
                # restart playback by rewinding capture
                video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False
                frames_displayed = 0
                print("⏮ Restarted video")

    finally:
        video.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("PLAYBACK SUMMARY")
        print("=" * 60)
        print(f"Video: {video_path.name}")
        print(f"Frames displayed: {frames_displayed}")
        print(f"Duration: {duration:.2f} seconds")
        print("=" * 60)


if __name__ == "__main__":
    main()
