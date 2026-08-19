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


class GazeUnavailableError(RuntimeError):
    """Raised by get_gaze_coords() when a real gaze sample couldn't be read
    and mouse-cursor simulation is disabled (AppConfig.GAZE_SIMULATION_ENABLED
    = False), so the caller doesn't silently receive fabricated data."""


class NoGazeTrackerError(RuntimeError):
    """Raised by EyeGazeTracker.__init__() when no Tobii 4C tracker is
    available and mouse-cursor simulation is disabled. A dedicated type
    (rather than a bare RuntimeError) so callers like main.py can catch this
    specific startup failure without also swallowing an unrelated
    RuntimeError from some other component."""


class EyeGazeTracker:
    """Manages the Tobii 4C eye tracker to get real-time gaze coordinates.

    Can optionally fall back to the mouse cursor position ("simulated" gaze)
    when the tracker package/hardware isn't available or a read fails, so
    the rest of the pipeline can be run and tested without the physical
    device attached - but only when AppConfig.GAZE_SIMULATION_ENABLED is
    explicitly set to True. With it left at the default False, missing or
    failed real gaze data raises instead of being silently faked, since
    mouse-simulated "gaze" mixed into a real session's statistics would be
    indistinguishable from genuine tracking data.
    """

    def __init__(self):
        self._gaze_center_x = -1
        self._gaze_center_y = -1
        self._gaze_bounding_box = None
        self._gaze_valid = False
        self._tracker = None
        self._last_gaze_time = time.time()
        self.is_simulated = False

        if app_config.TOBII_AVAILABLE and Tobii4cTracker is not None:
            try:
                self._tracker = Tobii4cTracker()
                print("Tobii 4C tracker initialized.")
            except Exception as e:
                print(f"Failed to initialize Tobii 4C tracker: {e}.")
                self._tracker = None
                app_config.TOBII_AVAILABLE = False

        if self._tracker is None:
            if not app_config.GAZE_SIMULATION_ENABLED:
                raise NoGazeTrackerError(
                    "No Tobii 4C tracker is available and gaze simulation is disabled "
                    "(AppConfig.GAZE_SIMULATION_ENABLED = False). Attach the tracker, or "
                    "explicitly set GAZE_SIMULATION_ENABLED = True in config.py for a "
                    "deliberate hardware-less test run (mouse cursor stands in for gaze)."
                )
            print("Tobii 4C tracker not available - gaze will be simulated (mouse cursor), "
                  "as GAZE_SIMULATION_ENABLED is True.")

    def get_gaze_coords(self):
        """Returns current gaze coordinates as (x, y, bounding_box) in
        absolute desktop coordinates. `bounding_box` is None for simulated
        (mouse-based) gaze.

        Raises GazeUnavailableError if a real tracker sample can't be read
        and GAZE_SIMULATION_ENABLED is False - callers should treat that as
        "skip this frame", not crash the whole session over one bad sample.
        """
        if self._tracker is not None:
            try:
                center: "TrackerInfo" = self._tracker.find_center()
            except Exception as e:
                print(f"Tobii tracker read failed: {e}.")
                center = None

            if center is not None and center.center_x is not None and center.center_y is not None:
                self._gaze_center_x = int(center.center_x)
                self._gaze_center_y = int(center.center_y)
                self._gaze_bounding_box = center.bounding_box
                self._gaze_valid = True
                self.is_simulated = False
                self._last_gaze_time = time.time()
                return (self._gaze_center_x, self._gaze_center_y, self._gaze_bounding_box)

            if not app_config.GAZE_SIMULATION_ENABLED:
                raise GazeUnavailableError(
                    "Tobii tracker read failed or returned an invalid sample, and gaze "
                    "simulation is disabled."
                )
            print("Tobii tracker read failed or invalid - falling back to simulated gaze for this sample.")

        # Reached only when GAZE_SIMULATION_ENABLED is True: either no
        # tracker was ever available (see __init__), or this sample's real
        # read failed/was invalid - simulate gaze via the mouse cursor.
        x, y = win32api.GetCursorPos()
        self._gaze_center_x, self._gaze_center_y = x, y
        self._gaze_bounding_box = None
        self._gaze_valid = True
        self.is_simulated = True
        self._last_gaze_time = time.time()
        return (x, y, None)
