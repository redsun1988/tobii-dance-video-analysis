import cv2
import numpy as np
# import face_recognition
from ultralytics import YOLO
from config import AppConfig

app_config = AppConfig()


class PoseEstimator:
    """Handles a YOLO-Pose model (YOLO26/YOLO11/YOLOv8) for human pose estimation and tracking."""

    # COCO keypoint indices, as produced by YOLO-Pose (same layout across
    # YOLOv8/YOLO11/YOLO26):
    # 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
    # 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
    # 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
    # 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    LEFT_HIP, RIGHT_HIP = 11, 12

    # Body regions built from several keypoints via a minimum-area rectangle
    # (used when 2+ points genuinely span an area, e.g. shoulders+hips for
    # the torso).
    MULTI_POINT_PARTS = {
        'head': [0, 1, 2, 3, 4],
        'torso': [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
    }

    # Limb regions built as a single straight segment between two joints
    # (e.g. shoulder -> elbow). Splitting each limb into upper/lower segments
    # (instead of one box across the whole limb) keeps the rotated box tight
    # even when the limb is bent, which a single shoulder-to-wrist box can't
    # represent.
    SEGMENT_PARTS = {
        'left_upper_arm': (LEFT_SHOULDER, 7),
        'left_forearm': (7, 9),
        'right_upper_arm': (RIGHT_SHOULDER, 8),
        'right_forearm': (8, 10),
        'left_thigh': (LEFT_HIP, 13),
        'left_shin': (13, 15),
        'right_thigh': (RIGHT_HIP, 14),
        'right_shin': (14, 16),
    }

    # A bare line segment (or a single point) has no inherent width, and a
    # multi-point group can degenerate to one when only 2 keypoints are
    # confident (e.g. torso with only the shoulders visible). In both cases
    # the box width is synthesized as a fraction of the person's own scale
    # (shoulder width, see `_person_scale`) instead of a fixed pixel value,
    # so it grows/shrinks with how large the person appears in frame.
    PART_WIDTH_RATIO = {
        'head': 0.9,
        'torso': 1.05,
        'left_upper_arm': 0.34, 'right_upper_arm': 0.34,
        'left_forearm': 0.26, 'right_forearm': 0.26,
        'left_thigh': 0.40, 'right_thigh': 0.40,
        'left_shin': 0.30, 'right_shin': 0.30,
    }

    MIN_PART_WIDTH_PX = 8.0
    SEGMENT_LENGTH_PAD_RATIO = 0.15  # extend segment ends past the joints by this fraction of its length
    MULTI_POINT_PAD_RATIO = 0.12     # pad head/torso rectangles by this fraction of person_scale

    # Connections for drawing skeletons (from Ultralytics YOLO-Pose example)
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
        print(f"Loading YOLO-Pose model: {model_name}...")
        self.model = YOLO(model_name)
        print("YOLO-Pose model loaded.")
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

                body_parts_boxes = self._build_body_part_boxes(keypoints, box, app_config.MIN_POSE_CONFIDENCE)

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

    @classmethod
    def _build_body_part_boxes(cls, keypoints, box, min_conf):
        """
        Builds rotated rectangles that closely follow the position and
        orientation of each body part, based on its keypoints.

        Returns a dict of part_name -> (a, b, width) | None, where a, b are
        the endpoints of the rectangle's central axis and width is measured
        perpendicular to that axis - the format expected by
        JavelinThrower.is_point_in_rotated_rect(point, a, b, width).
        """
        person_scale = cls._person_scale(keypoints, min_conf, box)
        body_parts_boxes = {}

        for part_name, indices in cls.MULTI_POINT_PARTS.items():
            pts = [keypoints[idx, :2] for idx in indices if keypoints[idx, 2] > min_conf]
            body_parts_boxes[part_name] = cls._multi_point_box(
                pts, person_scale, cls.PART_WIDTH_RATIO[part_name]
            )

        for part_name, (idx_a, idx_b) in cls.SEGMENT_PARTS.items():
            if keypoints[idx_a, 2] > min_conf and keypoints[idx_b, 2] > min_conf:
                width = max(person_scale * cls.PART_WIDTH_RATIO[part_name], cls.MIN_PART_WIDTH_PX)
                body_parts_boxes[part_name] = cls._segment_box(
                    keypoints[idx_a, :2], keypoints[idx_b, :2], width, cls.SEGMENT_LENGTH_PAD_RATIO
                )
            else:
                # Both joints defining this segment must be confident - a
                # single visible endpoint isn't enough to know the segment's
                # orientation, so skip it rather than guess.
                body_parts_boxes[part_name] = None

        return body_parts_boxes

    @classmethod
    def _person_scale(cls, keypoints, min_conf, box):
        """
        A per-person size reference (in pixels) used to scale body-part box
        widths proportionally to how large the person appears in the frame,
        instead of a fixed pixel padding. Prefers shoulder width, then hip
        width, then falls back to a fraction of the person's bounding box.
        """
        ls, rs = keypoints[cls.LEFT_SHOULDER], keypoints[cls.RIGHT_SHOULDER]
        if ls[2] > min_conf and rs[2] > min_conf:
            shoulder_width = float(np.linalg.norm(ls[:2] - rs[:2]))
            if shoulder_width > 1e-3:
                return shoulder_width

        lh, rh = keypoints[cls.LEFT_HIP], keypoints[cls.RIGHT_HIP]
        if lh[2] > min_conf and rh[2] > min_conf:
            hip_width = float(np.linalg.norm(lh[:2] - rh[:2]))
            if hip_width > 1e-3:
                return hip_width * 1.15  # hips are typically a bit narrower than shoulders

        x1, y1, x2, y2 = box
        return max(1.0, float(x2 - x1) * 0.35)

    @classmethod
    def _segment_box(cls, p1, p2, width, length_pad_ratio):
        """Builds a rotated rectangle around the straight segment p1->p2."""
        a = np.array(p1, dtype=np.float32)
        b = np.array(p2, dtype=np.float32)
        direction = b - a
        length = float(np.linalg.norm(direction))

        if length < 1e-6:
            half = width / 2.0
            return (tuple((a - [half, 0]).tolist()), tuple((a + [half, 0]).tolist()), float(width))

        unit = direction / length
        pad = max(length * length_pad_ratio, width * 0.25)
        a2 = a - unit * pad
        b2 = b + unit * pad
        return (tuple(a2.tolist()), tuple(b2.tolist()), float(width))

    @classmethod
    def _multi_point_box(cls, pts, person_scale, fallback_width_ratio):
        """
        Builds a rotated rectangle covering a group of keypoints (e.g. head
        or torso). With 3+ points the true minimum-area rectangle is used
        (so it follows the actual spread/orientation of the points); with
        fewer points there isn't enough information for a real rectangle, so
        a proportional width is synthesized instead.
        """
        if len(pts) == 0:
            return None

        if len(pts) == 1:
            width = max(person_scale * fallback_width_ratio, cls.MIN_PART_WIDTH_PX)
            return cls._segment_box(pts[0], pts[0], width, cls.SEGMENT_LENGTH_PAD_RATIO)

        if len(pts) == 2:
            width = max(person_scale * fallback_width_ratio, cls.MIN_PART_WIDTH_PX)
            return cls._segment_box(pts[0], pts[1], width, cls.SEGMENT_LENGTH_PAD_RATIO)

        pts_arr = np.array(pts, dtype=np.float32)
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

        pad = max(person_scale * cls.MULTI_POINT_PAD_RATIO, cls.MIN_PART_WIDTH_PX * 0.5)
        direction = b - a
        length = np.linalg.norm(direction)
        if length > 1e-6:
            unit = direction / length
            a = a - unit * pad
            b = b + unit * pad
        width = max(width + pad * 2, cls.MIN_PART_WIDTH_PX)

        return (tuple(a), tuple(b), float(width))

    # --- Video analysis caching (see database.py) -------------------------------

    @classmethod
    def people_data_to_cache_rows(cls, people_data, content_rect):
        """
        Converts one frame's process_frame() output into JSON-safe,
        resolution-independent rows for the video analysis cache: box/body-
        part endpoint coordinates are stored as fractions of `content_rect`
        (x0, y0, w, h) - the sub-rectangle of this frame actually occupied
        by the video's own content, not necessarily the whole captured
        frame (see video_geometry.letterboxed_content_rect). Normalizing
        against the content rect rather than the raw frame is what makes a
        cached video's layout stay valid however its capture window is
        later moved, resized, or shown on a different monitor, AND lets a
        live window-capture session and a headless direct file decode (see
        video_precomputer.py, where content_rect is simply the whole
        decoded frame) share the same coordinate space.

        A rotated box's `width` is its perpendicular thickness, not tied to
        a single axis, so it can't be split into independent x/y scale
        factors the way endpoint coordinates can; it's normalized against
        the content rect's height as a single reference axis, which is
        exact whenever the aspect ratio is preserved and only mildly
        approximate otherwise.

        Returns [] for a degenerate (zero-width or zero-height) content
        rect - e.g. a minimized player window - rather than raising, since
        that's just one sample worth silently dropping from the cache.
        """
        x0, y0, content_w, content_h = content_rect
        if content_w <= 0 or content_h <= 0:
            return []

        rows = []
        for person in people_data:
            x1, y1, x2, y2 = person['box']
            box_frac = [
                (x1 - x0) / content_w, (y1 - y0) / content_h,
                (x2 - x0) / content_w, (y2 - y0) / content_h,
            ]

            body_parts_frac = {}
            for part_name, rotated_box in person['body_parts'].items():
                if rotated_box is None:
                    body_parts_frac[part_name] = None
                    continue
                (ax, ay), (bx, by), width = rotated_box
                body_parts_frac[part_name] = [
                    (ax - x0) / content_w, (ay - y0) / content_h,
                    (bx - x0) / content_w, (by - y0) / content_h,
                    width / content_h,
                ]

            rows.append({
                'track_id': int(person['id']),
                'box': box_frac,
                'body_parts': body_parts_frac,
            })
        return rows

    @classmethod
    def cache_rows_to_people_data(cls, rows, content_rect):
        """
        Inverse of people_data_to_cache_rows(): reconstructs a
        process_frame()-shaped people_data list (only the 'id', 'box' and
        'body_parts' keys - the only ones any downstream consumer reads)
        from cached fractional coordinates, scaled to the current frame's
        `content_rect` (x0, y0, w, h).

        Returns [] for a degenerate (zero-width or zero-height) content
        rect - e.g. a momentarily minimized player window during cache
        replay - rather than collapsing every position to a single point,
        mirroring people_data_to_cache_rows' same-situation behavior.
        """
        x0, y0, content_w, content_h = content_rect
        if content_w <= 0 or content_h <= 0:
            return []

        people_data = []
        for row in rows:
            x1f, y1f, x2f, y2f = row['box']
            box = np.array([
                x0 + x1f * content_w, y0 + y1f * content_h,
                x0 + x2f * content_w, y0 + y2f * content_h,
            ])

            body_parts = {}
            for part_name, part in row['body_parts'].items():
                if part is None:
                    body_parts[part_name] = None
                    continue
                axf, ayf, bxf, byf, wf = part
                body_parts[part_name] = (
                    (x0 + axf * content_w, y0 + ayf * content_h),
                    (x0 + bxf * content_w, y0 + byf * content_h),
                    wf * content_h,
                )

            people_data.append({'id': row['track_id'], 'box': box, 'body_parts': body_parts})
        return people_data

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
