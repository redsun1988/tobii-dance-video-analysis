"""
Geometry shared between live gaze sessions (pose_gaze_application.py) and
headless cache precomputation (video_precomputer.py), so both normalize
cached positions into the exact same coordinate space and can therefore
reuse each other's cache - see database.py's `pose_cache_capture_mode`.
"""

import cv2


def read_video_metadata(filename):
    """Returns (duration_s, native_width, native_height) read directly from
    the video file - used both to size the live capture loop and to compute
    a captured frame's letterbox-corrected content rect (both callers need
    the video's own resolution for that, not just its duration)."""
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video file to read its metadata: {filename}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Could not determine duration of video file: {filename}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not determine native resolution of video file: {filename}")
    return frame_count / fps, int(width), int(height)


def letterboxed_content_rect(capture_w, capture_h, video_w, video_h):
    """
    Returns (x0, y0, w, h): the sub-rectangle of a `capture_w` x `capture_h`
    frame actually occupied by the video's content, assuming the player
    scales the video to fit the frame while preserving its aspect ratio
    (letterboxing/pillarboxing to black bars where it doesn't match) - the
    default behavior of Windows' built-in video player and most others.

    A live capture of the player's window client area generally isn't
    exactly the video's own pixels: if the window's aspect ratio doesn't
    match the video's, part of that captured frame is player letterbox
    bars, not video content. Normalizing cached coordinates against the
    raw captured frame (as done before this existed) silently drifts once
    the window is resized to a different aspect ratio. Normalizing against
    this content rect instead keeps coordinates meaningful, and - just as
    importantly - matches the coordinate space a headless direct-decode of
    the video file naturally produces (a decoded frame has no letterbox
    bars by construction, so its "content rect" is always the whole frame).

    Degenerates to the whole capture frame if either dimension is zero or
    negative, rather than raising, so a transient bad frame just doesn't
    get a correction applied instead of crashing the caller.
    """
    if capture_w <= 0 or capture_h <= 0 or video_w <= 0 or video_h <= 0:
        return (0.0, 0.0, float(capture_w), float(capture_h))

    capture_aspect = capture_w / capture_h
    video_aspect = video_w / video_h

    if capture_aspect > video_aspect:
        # Capture is relatively wider than the video - pillarboxed (bars on
        # left/right), content spans the full height.
        content_h = float(capture_h)
        content_w = content_h * video_aspect
        x0 = (capture_w - content_w) / 2.0
        y0 = 0.0
    else:
        # Capture is relatively taller than (or matches) the video -
        # letterboxed (bars on top/bottom), content spans the full width.
        content_w = float(capture_w)
        content_h = content_w / video_aspect
        x0 = 0.0
        y0 = (capture_h - content_h) / 2.0

    return (x0, y0, content_w, content_h)
