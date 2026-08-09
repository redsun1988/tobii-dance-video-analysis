import time
from config import AppConfig
from CustomTobii4cTracker import Tobii4cTracker, TrackerInfo

app_config = AppConfig()

class EyeGazeTracker:
    """Manages the Tobii 4C eye tracker to get real-time gaze coordinates."""
    def __init__(self):
        self._gaze_center_x = -1
        self._gaze_center_y = -1
        self._gaze_valid = False
        self._tracker = None
        self._last_gaze_time = time.time()
        
        if app_config.TOBII_AVAILABLE:
            try:
                self._tracker = Tobii4cTracker()
                print("Tobii 4C tracker initialized.")
            except Exception as e:
                print(f"Failed to initialize Tobii 4C tracker: {e}. Gaze tracking will be simulated.")
                self._tracker = None

                app_config.TOBII_AVAILABLE = False
        else:
            print("Tobii 4C tracker not available, gaze will be simulated.")

    def get_gaze_coords(self):
        """Returns current gaze coordinates (center_x, center_y) if valid, otherwise simulates."""
        center: TrackerInfo = self._tracker.find_center()
        if center.center_x is not None and center.center_y is not None:
            self._gaze_center_x = int(center.center_x)
            self._gaze_center_y = int(center.center_y)
            self._gaze_bounding_box = center.bounding_box

            self._gaze_valid = True
            self._last_gaze_time = time.time()
            print((self._gaze_center_x, self._gaze_center_y, self._gaze_bounding_box))
            return (self._gaze_center_x, self._gaze_center_y, self._gaze_bounding_box)
        else:
            self._gaze_valid = False
            return (0, 0, None)