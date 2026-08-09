import time
import subprocess
import numpy as np
from PIL import ImageGrab
from moviepy import VideoFileClip
from gaze_analyzer import GazeAnalyzer
from pose_estimator import PoseEstimator
from eye_gaze_tracker import EyeGazeTracker

class PoseGazeApplication:
    """Orchestrates the gaze tracking, pose estimation, and analysis."""
    def __init__(self):
        self.eye_tracker = EyeGazeTracker()
        self.pose_estimator = PoseEstimator()
        self.gaze_analyzer = GazeAnalyzer()

        self.fps = 30
        self.gaze_analyzer.set_frame_rate(self.fps)
        self.frame_idx = 0
        self._gaze_bounding_box = None

    def get_videofile_duration(self, filename):
        clip = VideoFileClip(filename)
        duration = clip.duration
        return duration

    async def run(self, video_path: str):
        """Starts the main application loop."""
        # Start the process
        p = subprocess.Popen(["start", video_path], shell=True)
        video_duration = self.get_videofile_duration(video_path)
        start_time = time.time()   # Получаем текущее время в секундах


        while (time.time()-start_time) < video_duration:  # Проверяем, сколько времени прошло с момента запуска метода run
            try:
                # Capture the entire screen
                pil_image = ImageGrab.grab()

                # Convert RGB to BGR
                frame = np.array(pil_image)[:, :, ::-1].copy()

                # Get gaze coordinates
                absolyte_gaze_x, absolyte_gaze_y, gaze_bounding_box = self.eye_tracker.get_gaze_coords()
                gaze_coords = (absolyte_gaze_x, absolyte_gaze_y)

                # Pose estimation and tracking
                people_data = self.pose_estimator.process_frame(frame)

                # Analyze gaze
                self.gaze_analyzer.analyze_frame(self.frame_idx, people_data, gaze_coords)

                for person in people_data:
                    # Highlight gazed body part
                    gaze_events_for_frame = self.gaze_analyzer.gaze_history.get(self.frame_idx, [])
                    for event in gaze_events_for_frame:
                        if event['gazed_person_id'] == person['id'] and event['gazed_body_part'] is not None:
                            part_box = person['body_parts'][event['gazed_body_part']]
                            if part_box:
                                
                                print(f"you are looking at person {person["faceface_id"]} and {event['gazed_body_part'].replace('_', ' ').title()}")

                self.frame_idx += 1
            except KeyboardInterrupt:
                break

        # Generate and display statistics
        stats = self.gaze_analyzer.generate_statistics()
        self.gaze_analyzer.display_statistics(stats)
        self.gaze_analyzer.save_to_excel("test.xlsx")