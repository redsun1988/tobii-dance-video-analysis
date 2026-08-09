import os
import time

import numpy as np
import win32api
import win32con
from PIL import ImageGrab
import cv2

from config import AppConfig
from gaze_analyzer import GazeAnalyzer
from pose_estimator import PoseEstimator
from eye_gaze_tracker import EyeGazeTracker
from vlm_attention_probe import VlmAttentionProbe
from video_window import VideoPlayerWindow, WindowNotFoundError
from javelin_thrower import JavelinThrower

app_config = AppConfig()


class PoseGazeApplication:
    """Orchestrates video playback window discovery, gaze tracking, pose
    estimation, gaze-to-body-part analysis, and the VLM attention probe."""

    def __init__(self):
        self.eye_tracker = EyeGazeTracker()
        self.pose_estimator = PoseEstimator()
        self.gaze_analyzer = GazeAnalyzer()
        self.vlm_probe = VlmAttentionProbe()
        self.frame_idx = 0

    @staticmethod
    def _get_video_duration(filename):
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open video file to read its duration: {filename}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Could not determine duration of video file: {filename}")
        return frame_count / fps

    @staticmethod
    def _virtual_screen_bbox():
        """Absolute bounds of the full multi-monitor desktop - used as the
        capture/coordinate reference when the video player window couldn't
        be located, so gaze-to-local coordinate math still works the same
        way as the windowed case."""
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        return (left, top, left + width, top + height)

    def _capture_frame(self, window):
        """
        Grabs the current video window content (or the whole desktop if the
        window couldn't be located / has vanished). Returns
        (frame_bgr, capture_bbox_absolute) - capture_bbox is always a real
        absolute (left, top, right, bottom), so downstream coordinate math
        doesn't need to special-case the fallback.
        """
        bbox = None
        if window is not None and window.is_alive():
            try:
                bbox = window.get_client_bbox_absolute()
            except WindowNotFoundError:
                bbox = None

        if bbox is None:
            bbox = self._virtual_screen_bbox()

        pil_image = ImageGrab.grab(bbox=bbox)
        frame = np.array(pil_image)[:, :, ::-1].copy()  # RGB -> BGR
        return frame, bbox

    @staticmethod
    def _to_local(gaze_absolute, capture_bbox):
        """Converts absolute desktop gaze coords into the captured frame's
        local coordinate space. Returns (local_xy, in_window)."""
        left, top, right, bottom = capture_bbox
        lx = gaze_absolute[0] - left
        ly = gaze_absolute[1] - top
        in_window = 0 <= lx <= (right - left) and 0 <= ly <= (bottom - top)
        return (lx, ly), in_window

    @staticmethod
    def _crop_box_for_hint(crop_hint):
        """
        Picks an axis-aligned crop region (x1, y1, x2, y2) around a
        newly-fixated target, for the VLM probe:
          - a specific body part -> tight axis-aligned bounds of its rotated
            box, with a small margin
          - a person with no matched part -> a box around the gaze point,
            sized relative to the person's own box
          - background -> a fixed-size box centered on the gaze point
        """
        kind = crop_hint['kind']
        gx, gy = crop_hint['gaze_point']

        if kind == 'part':
            a, b, width = crop_hint['rotated_box']
            margin = max(width * 0.25, 6.0)
            return JavelinThrower.rotated_rect_axis_aligned_bounds(a, b, width, margin=margin)

        if kind == 'person':
            x1, y1, x2, y2 = crop_hint['person_box']
            half_w = max((x2 - x1) * app_config.PERSON_CROP_RATIO, 20) / 2.0
            half_h = max((y2 - y1) * app_config.PERSON_CROP_RATIO, 20) / 2.0
            return (gx - half_w, gy - half_h, gx + half_w, gy + half_h)

        half = app_config.BACKGROUND_CROP_SIZE_PX / 2.0
        return (gx - half, gy - half, gx + half, gy + half)

    def run(self, video_path: str):
        """Launches the video, tracks its player window, and analyzes gaze
        against pose estimation until playback ends."""
        try:
            window = VideoPlayerWindow(min_size_px=app_config.WINDOW_MIN_SIZE_PX)
            window.launch_and_locate(video_path, timeout=app_config.WINDOW_DETECT_TIMEOUT_S)
            print(f"Located video player window: '{window.get_title()}'")
        except WindowNotFoundError as e:
            print(f"Warning: {e} Falling back to whole-screen capture.")
            window = None

        video_duration = self._get_video_duration(video_path)
        start_time = time.time()
        last_time = start_time

        while (time.time() - start_time) < video_duration:
            try:
                if window is not None and not window.is_alive():
                    print("Video player window closed - stopping early.")
                    break

                frame, capture_bbox = self._capture_frame(window)

                gaze_x, gaze_y, _ = self.eye_tracker.get_gaze_coords()
                gaze_absolute = (gaze_x, gaze_y)
                gaze_local, in_window = self._to_local(gaze_absolute, capture_bbox)
                gaze_source = "simulated" if self.eye_tracker.is_simulated else "tobii"

                people_data = self.pose_estimator.process_frame(frame)

                now = time.time()
                dt = now - last_time
                last_time = now
                timestamp = now - start_time

                analysis = self.gaze_analyzer.analyze_frame(
                    self.frame_idx, timestamp, people_data, gaze_absolute, gaze_local, in_window, gaze_source, dt
                )

                if analysis.is_new_fixation:
                    label = analysis.target.label
                    print(f"New fixation: {label}")
                    if self.vlm_probe.should_probe(label, timestamp):
                        crop_box = self._crop_box_for_hint(analysis.crop_hint)
                        self.vlm_probe.submit(frame, crop_box, label, self.frame_idx, timestamp)

                self.frame_idx += 1
            except KeyboardInterrupt:
                print("Interrupted by user - stopping early.")
                break

        self._finalize()

    def _finalize(self):
        print("Waiting for pending VLM attention-probe queries to finish...")
        self.vlm_probe.stop()

        stats = self.gaze_analyzer.generate_statistics()
        self.gaze_analyzer.display_statistics(stats)

        run_id = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(app_config.CSV_OUTPUT_DIR, run_id)
        os.makedirs(output_dir, exist_ok=True)

        self.gaze_analyzer.save_to_csv(output_dir, vlm_query_log=self.vlm_probe.query_log)
        self.gaze_analyzer.save_to_excel(os.path.join(output_dir, "gaze_statistics.xlsx"))
