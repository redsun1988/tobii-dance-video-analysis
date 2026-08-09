class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class AppConfig(metaclass=Singleton):
    def __init__(self):
        self.YOLO_MODEL_NAME  = 'yolov8n-pose.pt'
        self.MIN_POSE_CONFIDENCE  = 0.5   # Minimum confidence for a keypoint to be considered valid [1]
        self.DRAW_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to draw a keypoint/connection [1]

        # --- Video window discovery -------------------------------------------------
        # How long to wait for the OS-default video player window to appear after
        # launching the video file, before giving up and falling back to
        # whole-screen capture.
        self.WINDOW_DETECT_TIMEOUT_S = 15.0
        self.WINDOW_MIN_SIZE_PX = (200, 150)  # ignore tiny popups/toasts when guessing the player window

        # --- Saccade / fixation detection --------------------------------------------
        # A gaze jump is treated as a saccade if it exceeds either threshold.
        self.SACCADE_MIN_DISTANCE_PX = 80.0
        self.SACCADE_MIN_VELOCITY_PX_S = 600.0
        # Consecutive samples the gaze must stay on the same new target after a
        # saccade before we treat it as a confirmed fixation (debounces noisy landings).
        self.FIXATION_CONFIRM_SAMPLES = 2

        # --- Local VLM ("what caught your eye") attention probe ----------------------
        self.VLM_ENABLED = True
        self.VLM_MODEL_NAME = "qwen2.5vl:7b"  # vision-capable model already pulled locally; swap to "llava" once pulled
        self.VLM_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint
        self.VLM_API_KEY = "ollama"  # unused by Ollama but required by the OpenAI SDK
        self.VLM_PROMPT = (
            "This is a cropped frame from a dance video, taken from the region a viewer "
            "just looked at right after a sudden eye movement. In 1-2 short sentences, "
            "describe what is visible in the crop and what about it (motion, color, body "
            "part, contrast, position) could have attracted the viewer's attention."
        )
        # This local model runs CPU-only on this machine (no GPU acceleration
        # for Ollama here) - measured ~200s for a single small crop, so the
        # timeout needs a lot of headroom. The bounded queue (below) is what
        # actually keeps the capture loop responsive, not a short timeout.
        self.VLM_REQUEST_TIMEOUT_S = 240.0
        self.VLM_QUEUE_MAXSIZE = 2  # backlog before new probe requests are skipped (and logged as skipped)
        self.VLM_COOLDOWN_S = 4.0  # minimum time before re-probing the same target label
        self.VLM_JPEG_QUALITY = 85
        self.BACKGROUND_CROP_SIZE_PX = 240  # crop size when gaze lands on no detected person
        self.PERSON_CROP_RATIO = 0.4  # crop size (fraction of person box) when gaze is on a person but no specific part

        # --- Output -------------------------------------------------------------------
        self.CSV_OUTPUT_DIR = "output"

        try:
            import CustomTobii4cTracker
            self.TOBII_AVAILABLE = True
        except ImportError:
            print("Warning: CustomTobii4cTracker not found. Gaze tracking will be simulated (mouse cursor).")
            self.TOBII_AVAILABLE = False

        try:
            import ultralytics  # YOLOv8-Pose for Human Pose Estimation and Tracking
            self.YOLO_AVAILABLE  = True
        except ImportError:
            print("Error: ultralytics (YOLO) library not found. Pose estimation is unavailable.")
            self.YOLO_AVAILABLE = False

        try:
            import openai  # OpenAI-compatible client used to talk to the local Ollama server
            from openai import OpenAI
            ollama_key = self.VLM_API_KEY
            client_kwargs = {"base_url": self.VLM_BASE_URL, "api_key": ollama_key}
            probe_client = OpenAI(**client_kwargs)
            probe_client.models.list()
            self.OLLAMA_AVAILABLE = True
        except Exception as e:
            print(f"Warning: local Ollama server not reachable at {self.VLM_BASE_URL} ({e}). "
                  "Attention probing on new gaze targets will be disabled.")
            self.OLLAMA_AVAILABLE = False