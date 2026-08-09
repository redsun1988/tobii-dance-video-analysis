import sys

from video_window import enable_dpi_awareness
from config import AppConfig

app_config = AppConfig()
from pose_gaze_application import PoseGazeApplication

DEFAULT_VIDEO_PATH = "dancing_people.mp4"

if __name__ == "__main__":
    # Must happen before any window rects / screen grabs / cursor
    # coordinates are read, so they all agree on physical pixels on
    # HiDPI/multi-monitor setups.
    enable_dpi_awareness()

    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH

    if not app_config.TOBII_AVAILABLE:
        print("Tobii 4C tracker not available - gaze will be simulated using the mouse cursor.")

    PoseGazeApplication().run(video_path)
