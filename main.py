import sys

from video_window import enable_dpi_awareness
from config import AppConfig

app_config = AppConfig()
from pose_gaze_application import PoseGazeApplication
from eye_gaze_tracker import NoGazeTrackerError

DEFAULT_VIDEO_PATH = "dancing_people.mp4"

if __name__ == "__main__":
    # Must happen before any window rects / screen grabs / cursor
    # coordinates are read, so they all agree on physical pixels on
    # HiDPI/multi-monitor setups.
    enable_dpi_awareness()

    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH

    if not app_config.TOBII_AVAILABLE and app_config.GAZE_SIMULATION_ENABLED:
        print("Tobii 4C tracker not available - gaze will be simulated using the mouse cursor.")

    try:
        app = PoseGazeApplication()
    except NoGazeTrackerError as e:
        print(f"Error: {e}")
        sys.exit(1)

    app.run(video_path)
