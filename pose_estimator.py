import cv2
import uuid
import numpy as np
# import face_recognition
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
                
                # Calculate rotated boxes for body parts (follow the actual
                # orientation of the limb/torso instead of an axis-aligned box)
                body_parts_boxes = {}
                for part_name, kp_indices in self.body_part_keypoints.items():
                    rotated_box = self._calculate_rotated_box(
                        keypoints, kp_indices, app_config.MIN_POSE_CONFIDENCE
                    )
                    # rotated_box is either None or a tuple (a, b, width) where
                    # a, b are the endpoints of the box's central axis and
                    # width is the box width perpendicular to that axis.
                    body_parts_boxes[part_name] = rotated_box

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
    def _calculate_rotated_box(keypoints, indices, min_conf, padding=10):
        """
        Builds a rotated rectangle that closely follows the position and
        orientation of a body part, based on its keypoints.

        Returns a tuple (a, b, width) where:
            a, b   - endpoints of the rectangle's central axis (its "length"
                     direction, e.g. shoulder -> wrist for an arm)
            width  - rectangle width, measured perpendicular to the a-b axis

        This representation matches what
        JavelinThrower.is_point_in_rotated_rect(point, a, b, width) expects.

        Returns None if there aren't enough confident keypoints to build a box.
        """
        pts = []
        for idx in indices:
            if idx < len(keypoints) and keypoints[idx][2] > min_conf:  # Check confidence
                pts.append([keypoints[idx][0], keypoints[idx][1]])

        if len(pts) == 0:
            return None  # No confident keypoints to form a box

        if len(pts) == 1:
            # Only one reliable point - build a small square-ish box around it
            x, y = pts[0]
            a = (x - padding, y)
            b = (x + padding, y)
            return (a, b, float(padding * 2))

        pts_arr = np.array(pts, dtype=np.float32)

        # minAreaRect finds the minimum-area rectangle enclosing the points,
        # at whatever angle best fits them - this is what gives us rotation
        # that follows the limb/torso orientation instead of axis alignment.
        rect = cv2.minAreaRect(pts_arr)  # ((cx, cy), (w, h), angle)
        box_points = cv2.boxPoints(rect)  # 4 corners of the rectangle

        edge1 = np.linalg.norm(box_points[1] - box_points[0])
        edge2 = np.linalg.norm(box_points[2] - box_points[1])

        # Use the longer edge as the central axis (a-b), the shorter one
        # becomes the rectangle's width
        if edge1 >= edge2:
            a = (box_points[0] + box_points[3]) / 2.0
            b = (box_points[1] + box_points[2]) / 2.0
            width = edge2
        else:
            a = (box_points[0] + box_points[1]) / 2.0
            b = (box_points[3] + box_points[2]) / 2.0
            width = edge1

        # Small padding along the axis and across the width, similar to the
        # padding previously applied to the axis-aligned bounding box
        direction = b - a
        length = np.linalg.norm(direction)
        if length > 1e-6:
            unit = direction / length
            a = a - unit * padding
            b = b + unit * padding
        width += padding * 2

        return (tuple(a), tuple(b), float(width))
    
    def get_face_id(self, frame, box):
        x1, y1, x2, y2 = box
        cropped_frame = frame[y1:y2, x1:x2]
        
        # # Use the face_recognition library to find faces in the cropped frame
        # face_locations = face_recognition.face_locations(cropped_frame)
        
        # if len(face_locations) > 0:  # If a face is found
        #     face_encoding = face_recognition.face_encodings(cropped_frame, known_face_locations=face_locations)[0]
            
        #     # Check if this face encoding is already in our list of known faces
        #     for face, id in self.known_faces.items():
        #         match = face_recognition.compare_faces([face], face_encoding)
                
        #         if match[0]:  # If it's a known face
        #             return id  # Return the corresponding ID
            
        #     # If it's a new face, assign it a new unique ID and add to the list of known faces
        #     new_id = uuid.uuid4()
        #     self.known_faces[face_encoding] = new_id
        #     return new_id
        
        # else:  # If no face is found in the cropped frame, return None
        return None
        
    def get_age(self, frame, box): return None  # TODO: implement this method
    def get_gender(self, frame, box): return None  # TODO: implement this method
    def get_emotion(self, frame, box): return None  # TODO: implement this method