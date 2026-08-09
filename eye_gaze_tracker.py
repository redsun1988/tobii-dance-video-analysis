import time
import win32api
from config import AppConfig

app_config = AppConfig()

try:
    from CustomTobii4cTracker import Tobii4cTracker, TrackerInfo
except ImportError:
    # Package not installed at all - config.py already warned about this via
    # TOBII_AVAILABLE. Keep the names defined so the rest of this module can
    # still be imported (and simulated gaze still works) without it.
    Tobii4cTracker = None
    TrackerInfo = None


class EyeGazeTracker:
    """Manages the Tobii 4C eye tracker to get real-time gaze coordinates.

    Falls back to the mouse cursor position ("simulated" gaze) whenever the
    tracker package/hardware isn't available, so the rest of the pipeline can
    be run and tested without the physical device attached.
    """

    def __init__(self):
        self._gaze_center_x = -1
        self._gaze_center_y = -1
        self._gaze_bounding_box = None
        self._gaze_valid = False
        self._tracker = None
        self._last_gaze_time = time.time()
        self.is_simulated = True

        if app_config.TOBII_AVAILABLE and Tobii4cTracker is not None:
            try:
                self._tracker = Tobii4cTracker()
                self.is_simulated = False
                print("Tobii 4C tracker initialized.")
            except Exception as e:
                print(f"Failed to initialize Tobii 4C tracker: {e}. Gaze tracking will be simulated (mouse cursor).")
                self._tracker = None
                app_config.TOBII_AVAILABLE = False
        else:
            print("Tobii 4C tracker not available, gaze will be simulated (mouse cursor).")

    def get_gaze_coords(self):
        """Returns current gaze coordinates as (x, y, bounding_box) in
        absolute desktop coordinates. `bounding_box` is None for simulated
        (mouse-based) gaze.
        """
        if self._tracker is not None:
            try:
                center: "TrackerInfo" = self._tracker.find_center()
            except Exception as e:
                print(f"Tobii tracker read failed: {e}. Falling back to simulated gaze for this sample.")
                center = None

            if center is not None and center.center_x is not None and center.center_y is not None:
                self._gaze_center_x = int(center.center_x)
                self._gaze_center_y = int(center.center_y)
                self._gaze_bounding_box = center.bounding_box
                self._gaze_valid = True
                self.is_simulated = False
                self._last_gaze_time = time.time()
                return (self._gaze_center_x, self._gaze_center_y, self._gaze_bounding_box)

        # No tracker, tracker read failed, or tracker returned an invalid
        # sample (e.g. eyes off-screen) - simulate gaze via the mouse cursor.
        x, y = win32api.GetCursorPos()
        self._gaze_center_x, self._gaze_center_y = x, y
        self._gaze_bounding_box = None
        self._gaze_valid = True
        self.is_simulated = True
        self._last_gaze_time = time.time()
        return (x, y, None)
