from video_window import enable_dpi_awareness
from config import AppConfig

app_config = AppConfig()
from start_menu import StartMenu

if __name__ == "__main__":
    # Must happen before any window rects / screen grabs / cursor
    # coordinates are read, so they all agree on physical pixels on
    # HiDPI/multi-monitor setups.
    enable_dpi_awareness()

    if not app_config.TOBII_AVAILABLE and app_config.GAZE_SIMULATION_ENABLED:
        print("Tobii 4C tracker not available - gaze will be simulated using the mouse cursor.")

    StartMenu().run()
