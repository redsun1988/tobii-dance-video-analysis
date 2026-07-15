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
        
        try:
            import CustomTobii4cTracker
            self.TOBII_AVAILABLE = True
        except ImportError:
            print("Warning: CustomTobii4cTracker not found. Gaze tracking will be simulated.")
            self.TOBII_AVAILABLE = False
        
        try:
            import ultralytics  # YOLOv8-Pose for Human Pose Estimation and Tracking
            self.YOLO_AVAILABLE  = True
        except ImportError:
            print("Error: ultralytics (YOLO) library not found. Pose estimation is unavailable.")
            self.YOLO_AVAILABLE = False