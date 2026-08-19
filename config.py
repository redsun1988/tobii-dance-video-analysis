class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class AppConfig(metaclass=Singleton):
    def __init__(self):
        self.YOLO_MODEL_NAME  = 'yolo26n-pose.pt'
        self.MIN_POSE_CONFIDENCE  = 0.5   # Minimum confidence for a keypoint to be considered valid [1]
        self.DRAW_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to draw a keypoint/connection [1]

        # --- Gaze source --------------------------------------------------------------
        # Simulating gaze via the mouse cursor exists only to exercise the pipeline
        # without the physical Tobii 4C tracker attached. It must stay off by default:
        # in a real session it would silently produce meaningless statistics (mouse
        # position mistaken for actual gaze) instead of failing loudly. Flip this to
        # True only for a deliberate hardware-less test run.
        self.GAZE_SIMULATION_ENABLED = False
        # Throttle retries while a real tracker's gaze sample is unavailable (blink,
        # momentary tracking loss), so a prolonged hardware outage doesn't busy-spin
        # full-screen captures with zero delay between attempts.
        self.GAZE_UNAVAILABLE_RETRY_DELAY_S = 0.5

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
        self.VLM_MODEL_NAME = "muse-glimmer:latest"  # vision-capable model already pulled locally; swap to "llava" once pulled
        self.VLM_BASE_URL = "http://192.168.1.12:11434/v1"  # Ollama's OpenAI-compatible endpoint
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

        # --- Post-hoc person identity reconciliation ----------------------------------
        # Pose tracking fragments one real dancer into several track IDs when they're
        # occluded or leave/re-enter frame. After playback ends, this samples a few
        # crops per track ID and asks the local VLM to compare each of one ID's crops
        # against a reference crop of another ID; if a majority say "same person", the
        # two IDs are merged before final statistics are computed. Only ID pairs where
        # both IDs received at least one confirmed gaze fixation are compared (an ID
        # nobody looked at can't skew gaze stats), and IDs that were ever visible in
        # the same frame are never compared (they can't be the same person).
        self.IDENTITY_RECONCILE_ENABLED = True
        self.IDENTITY_RECONCILE_CROPS_PER_PERSON = 3  # sample crops kept per track ID for cross-ID comparison
        # Fraction of "same person" votes (exclusive) needed to merge. Votes are whole
        # crops, so the *effective* agreement bar is rounded up by
        # IDENTITY_RECONCILE_CROPS_PER_PERSON, not exactly this fraction - e.g. at the
        # current 3 crops, 0.5 actually requires 2/3 (~67%) agreement, not ~50%; it
        # would be 3/5 (60%) at 5 crops. That effective bar shifts whenever the crop
        # count changes, even though this threshold value doesn't.
        self.IDENTITY_RECONCILE_MAJORITY_THRESHOLD = 0.5
        self.IDENTITY_RECONCILE_PROMPT = (
            "These are two cropped photos from a dance video. The first photo is a "
            "reference image of one tracked person. The second photo may or may not "
            "show the same individual person - judge by clothing, hair, build and "
            "visible skin tone, not by pose or camera angle. Answer with a single "
            "word first, 'yes' or 'no', then a short reason."
        )

        # --- Post-hoc appearance estimation for every detected person ------------------
        # A single frame is an unreliable source for these judgments (motion blur, bad
        # angle, occlusion), so - like identity reconciliation above - this samples a
        # few crops per track ID while the video plays, then after playback asks the
        # local VLM to judge each crop independently, per attribute, and takes a
        # majority vote across those judgments. Every track ID that collected at least
        # one crop is judged, regardless of whether it ever received a gaze fixation.
        self.DEMOGRAPHICS_ENABLED = True
        self.DEMOGRAPHICS_CROPS_PER_PERSON = 2  # sampled crops per person for the majority vote
        # Fraction of an attribute's non-null votes (exclusive) a category needs to be
        # reported as that person's value; otherwise the attribute is left unresolved
        # (None / 'unknown') rather than reporting a bare plurality winner from a
        # near-tied vote.
        self.DEMOGRAPHICS_MAJORITY_THRESHOLD = 0.5
        self.DEMOGRAPHICS_AGE_CATEGORIES = ('child', 'teen', 'young_adult', 'adult', 'senior')
        self.DEMOGRAPHICS_GENDERS = ('male', 'female')
        self.DEMOGRAPHICS_BODY_BUILDS = ('slim', 'athletic', 'average', 'heavy')
        self.DEMOGRAPHICS_CLOTHING_COLORS = (
            'black', 'white', 'gray', 'red', 'orange', 'yellow', 'green', 'blue',
            'purple', 'pink', 'brown', 'multicolor',
        )
        self.DEMOGRAPHICS_PROMPT = (
            "This is a cropped photo of one person from a dance video. Estimate this "
            "person's approximate age category, apparent gender, body build, and the "
            "single most dominant color of their clothing, based on their visible face, "
            "build, hair and clothing. Respond on the first line with exactly four words "
            "separated by commas, in this order: "
            "1) age category from [child, teen, young_adult, adult, senior], "
            "2) gender from [male, female], "
            "3) body build from [slim, athletic, average, heavy], "
            "4) dominant clothing color from [black, white, gray, red, orange, yellow, "
            "green, blue, purple, pink, brown, multicolor]. "
            "For example: 'young_adult, female, athletic, black'. Then give a short "
            "reason on the next line."
        )

        # --- Output -------------------------------------------------------------------
        self.CSV_OUTPUT_DIR = "output"

        try:
            import CustomTobii4cTracker
            self.TOBII_AVAILABLE = True
        except ImportError:
            self.TOBII_AVAILABLE = False
            if self.GAZE_SIMULATION_ENABLED:
                print("Warning: CustomTobii4cTracker not found. Gaze tracking will be simulated (mouse cursor).")
            else:
                print("Warning: CustomTobii4cTracker not found, and GAZE_SIMULATION_ENABLED is False - "
                      "a real Tobii 4C tracker will be required to start.")

        try:
            import ultralytics  # YOLO-Pose for Human Pose Estimation and Tracking
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