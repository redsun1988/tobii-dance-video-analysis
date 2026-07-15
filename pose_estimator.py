import cv2
import uuid
import numpy as np
import face_recognition
from ultralytics import YOLO
from config import AppConfig

app_config = AppConfig()

class PoseEstimator:
    """Handles YOLOv8-Pose model for human pose estimation and tracking."""
    
    # Define keypoint indices for body parts based on YOLOv8-Pose output (similar to MediaPipe)
    body_part_keypoints = {
        'head': [1, 2],
        'torso': [5, 7, 9],
        'left_arm': [6, 8, 10],
        'right_arm': [7, 9, 11],
        'left_leg': [11, 13, 15],
        'right_leg': [12, 14, 16]
    }
    
    # Connections for drawing skeletons (from YOLOv8-Pose example)
    pose_connections = [
        (5, 7), (7, 9),       # Left arm
        (6, 8), (8, 10),      # Right arm
        (11, 13), (13, 15),   # Left leg
        (12, 14), (14, 16),   # Right leg
        (5, 6), (5, 11), (6, 12),   # Torso
    ]
    
    def __init__(self, model_name=app_config.YOLO_MODEL_NAME):
        if not app_config.YOLO_AVAILABLE:
            raise ImportError("YOLO library is not available.")
        print(f"Loading YOLOv8-Pose model: {model_name}...")
        self.model = YOLO(model_name)
        print("YOLOv8-Pose model loaded.")
        self.known_faces = {}  # mapping from face encoding to unique ID
    
    def process_frame(self, frame):
        """
        Processes a single video frame to detect and track human poses.
        Returns a list of detected people with their IDs, bounding boxes, and keypoints.
        """
        # The main part that slow down fps
        results = self.model.track(frame, persist=True, classes=0, conf=app_config.MIN_POSE_CONFIDENCE, verbose=False)[0]  # classes=0 for 'person'
        people_data = []
        
        if results.boxes and results.boxes.id is not None and results.keypoints:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.int().cpu().tolist()
            keypoints_data = results.keypoints.data.cpu().numpy()  # [num_people, num_keypoints, 3 (x, y, conf)]
            
            for i in range(len(track_ids)):
                person_id = track_ids[i]
                box = boxes[i]  # [x1, y1, x2, y2]
                keypoints = keypoints_data[i]  # [17 keypoints, each with (x, y, conf)]
                
                # Calculate bounding boxes for body parts
                frame_height, frame_width = frame.shape[:2]
                body_parts_boxes = {}
                for part_name, kp_indices in self.body_part_keypoints.items():
                    bbox = self._calculate_bounding_box(keypoints, kp_indices, frame_width, frame_height)
                    # rotated_bbox = self._get_rotated_box(frame, self._get_central_line(keypoints))
                    rotated_bbox = None
                    body_parts_boxes[part_name] = (bbox, rotated_bbox)

                face_id = self.get_face_id(frame, box)
                emotion = self.get_emotion(frame, box)
                age = self.get_age(frame, box)
                gender = self.get_gender(frame, box)

                people_data.append({
                     'id': person_id,
                     'box': box,
                     'keypoints': keypoints,
                     'body_parts': body_parts_boxes,
                     'face_id': face_id,
                     'emotion': emotion,
                     'age': age,
                     'gender': gender
                    })
        return people_data
    
    @staticmethod
    def _calculate_bounding_box(keypoints, indices, frame_width, frame_height, min_conf=app_config.MIN_POSE_CONFIDENCE):
        """Calculates a bounding box around specified keypoints."""
        x_coords = []
        y_coords = []
        for idx in indices:
            if idx < len(keypoints) and keypoints[idx][2] > min_conf:  # Check confidence
                x_coords.append(keypoints[idx][0])
                y_coords.append(keypoints[idx][1])
        
        if not x_coords or not y_coords:
            return None  # No confident keypoints to form a box
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add a small padding and ensure coordinates are within frame boundaries
        padding = 10  # pixels
        x1 = max(0, int(min_x) - padding)
        y1 = max(0, int(min_y) - padding)
        x2 = min(frame_width - 1, int(max_x) + padding)
        y2 = min(frame_height - 1, int(max_y) + padding)
        
        return (x1, y1, x2, y2)
    
    @staticmethod
    def _get_central_line(keypoints):
        """Calculate a central line from two given keypoints."""
        
        # x1 = (keypoints[0][0] + keypoints[1][0]) / 2
        # y1 = (keypoints[0][1] + keypoints[1][1]) / 2

        # x2 = (keypoints[2][0] + keypoints[3][0]) / 2
        # y2 = (keypoints[2][1] + keypoints[3][1]) / 2
        x1, x2 = keypoints[0][0], keypoints[1][0]
        y1, y2 = keypoints[2][1], keypoints[3][1]
        central_line = ((x1, y1), (x2, y2))
        return central_line
    
    @staticmethod
    def _get_rotated_box(frame, central_line):
        """Calculate a rotated bounding box from the central line."""
        
        height, width = frame.shape[:2]
        distance = np.sqrt((central_line[1][0] - central_line[0][0])**2 + (central_line[1][1] - central_line[0][1])**2)
        
        # Generate a box with the same size as the central line but rotated
        box = cv2.boxPoints(((central_line[0], central_line[1], distance/4), 0))
        
        # Ensure the bounding box is within the frame boundaries
        for i in range(4):
            if box[i][0] < 0: box[i][0] = 0
            elif box[i][0] > width - 1: box[i][0] = width - 1
            
            if box[i][1] < 0: box[i][1] = 0
            elif box[i][1] > height - 1: box[i][1] = height - 1
        
        return np.int0(box)
    
    def get_face_id(self, frame, box):
        x1, y1, x2, y2 = box
        cropped_frame = frame[y1:y2, x1:x2]
        
        # Use the face_recognition library to find faces in the cropped frame
        face_locations = face_recognition.face_locations(cropped_frame)
        
        if len(face_locations) > 0:  # If a face is found
            face_encoding = face_recognition.face_encodings(cropped_frame, known_face_locations=face_locations)[0]
            
            # Check if this face encoding is already in our list of known faces
            for face, id in self.known_faces.items():
                match = face_recognition.compare_faces([face], face_encoding)
                
                if match[0]:  # If it's a known face
                    return id  # Return the corresponding ID
            
            # If it's a new face, assign it a new unique ID and add to the list of known faces
            new_id = uuid.uuid4()
            self.known_faces[face_encoding] = new_id
            return new_id
        
        else:  # If no face is found in the cropped frame, return None
            return None
        
    def get_age(self, frame, box): return None  # TODO: implement this method
    def get_gender(self, frame, box): return None  # TODO: implement this method
    def get_emotion(self, frame, box): return None  # TODO: implement this method