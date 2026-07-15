from config import AppConfig

app_config = AppConfig()
from pose_gaze_application import PoseGazeApplication

if __name__ == "__main__":
    # Ensure asyncio is available for Tobii tracking if needed
    if app_config.TOBII_AVAILABLE:
        import asyncio
        video_path: str = "dancing_people.mp4"

        asyncio.run(PoseGazeApplication().run(video_path))
    else:
        print(f"The TOBII_AVAILABLE variable is {app_config.TOBII_AVAILABLE}")
        print("Exit this program")
