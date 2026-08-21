"""Webcam discovery: enumerates the local video-capture devices attached to
this machine so the start menu's settings screen can let the user pick
which one records the tested viewer's face during a session - see
emotion_tracker.py, which opens whatever index is picked here.

OpenCV can't report a camera's friendly device name through the DirectShow
backend the way Windows itself can, so each entry is only distinguishable by
its index and the resolution the device reports opening at - good enough to
tell "the built-in laptop camera" from "the USB webcam" apart on hardware
with more than one.
"""

import time

import cv2

from config import AppConfig

app_config = AppConfig()

# Much faster to probe/open than OpenCV's default (CAP_MSMF) backend on
# Windows, which can take several seconds per missing index.
CAPTURE_BACKEND = cv2.CAP_DSHOW

# How many back-to-back failed reads preview_camera() tolerates before
# concluding the camera itself is gone (unplugged, claimed by another app)
# rather than just a transient DSHOW hiccup - see preview_camera().
MAX_CONSECUTIVE_READ_FAILURES = 60


def list_available_cameras(max_index=8):
    """Probes camera indices 0..max_index-1, opening each briefly to check
    it actually produces a frame (an index that's present but unopenable -
    e.g. already claimed by another application - is skipped). Returns a
    list of {'index': int, 'width': int, 'height': int} for every working
    camera, in index order."""
    cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, CAPTURE_BACKEND)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            cameras.append({'index': index, 'width': width, 'height': height})
        finally:
            cap.release()
    return cameras


def _load_emotion_analyzer():
    """Lazily imports DeepFace (same reasoning as emotion_tracker.py: a
    process that never previews a camera shouldn't pay TensorFlow's import
    cost). Returns None if the library isn't installed, so the preview
    still works - just without face boxes/emotion labels."""
    if not app_config.DEEPFACE_AVAILABLE:
        return None
    from deepface import DeepFace
    return DeepFace.analyze


def _detect_faces(analyze, frame_bgr):
    """Runs DeepFace's detector+emotion classifier on one frame and returns
    a list of {'box': (x, y, w, h), 'emotion': str} for every face actually
    found. Mirrors the face_confidence filtering EmotionTracker._process_frame
    uses: with enforce_detection=False, a faceless frame still comes back as
    one result covering the whole frame with face_confidence == 0, which is
    excluded here rather than drawn as a bogus box."""
    try:
        results = analyze(
            frame_bgr,
            actions=['emotion'],
            detector_backend=app_config.EMOTION_DETECTOR_BACKEND,
            enforce_detection=False,
            silent=True,
        )
    except Exception:
        return []

    faces = []
    for r in results:
        if r.get('face_confidence', 0) <= 0:
            continue
        region = r.get('region') or {}
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
        if w <= 0 or h <= 0:
            continue
        faces.append({'box': (x, y, w, h), 'emotion': r.get('dominant_emotion', '')})
    return faces


def preview_camera(index, window_title=None):
    """Opens a live preview window for the given camera index, so the user
    can see what the selected webcam actually frames before starting a
    session, with a bounding box and recognized emotion drawn over every
    detected face (same DeepFace classifier as emotion_tracker.py, polled
    at AppConfig.EMOTION_SAMPLE_INTERVAL_S - running it on every single
    frame would make the preview noticeably laggy for no real benefit,
    since a face's expression doesn't change that fast). Pressing S inside
    the window opens the camera driver's own DirectShow properties dialog
    (white balance, brightness, contrast, exposure, focus, etc.) - reusing
    the dialog the driver ships with is far more reliable than
    reimplementing sliders for these controls ourselves, since OpenCV has
    no way to query a given camera's real min/max/step for each property,
    but the driver's own dialog always does. Blocks until the user closes
    the preview (Esc/Q or the window's close button). Returns False if the
    camera could not be opened.
    """
    title = window_title or f"Cam_{index}"
    cap = cv2.VideoCapture(index, CAPTURE_BACKEND)
    if not cap.isOpened():
        cap.release()
        return False

    analyze = _load_emotion_analyzer()
    faces = []
    next_analysis_at = 0.0
    shown_at = None

    try:
        # No separate namedWindow() call: pre-creating an empty WINDOW_NORMAL
        # window and only later imshow()-ing a differently-sized frame into
        # it makes some Windows OpenCV builds silently recreate a second HWND
        # to fit the new size instead of resizing the first one, leaving a
        # blank "ghost" window behind that never receives key events and
        # doesn't close. Letting the first imshow() create the window - at
        # the real frame size, from the very first frame - avoids that
        # resize/recreate path entirely.
        consecutive_failures = 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # DSHOW webcams routinely drop a frame transiently (e.g. right
                # after the loop stalls on a slow DeepFace call) - treating
                # that as "camera closed" ended the preview after a single
                # bad frame. Only give up once failures persist long enough
                # to mean the camera was actually unplugged/lost, matching
                # how EmotionTracker._worker_loop treats a bad read as a
                # skippable poll rather than fatal.
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    break
                cv2.waitKey(1)
                continue
            consecutive_failures = 0

            if analyze is not None and time.monotonic() >= next_analysis_at:
                faces = _detect_faces(analyze, frame)
                next_analysis_at = time.monotonic() + app_config.EMOTION_SAMPLE_INTERVAL_S

            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, face['emotion'], (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                )

            cv2.imshow(title, frame)
            if shown_at is None:
                shown_at = time.monotonic()
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('s'), ord('S')):
                # CAP_PROP_SETTINGS pops up the device's own DirectShow
                # properties dialog on Windows; the call blocks until the
                # user closes that dialog, then the preview loop resumes
                # and shows frames with whatever settings were changed.
                cap.set(cv2.CAP_PROP_SETTINGS, 1)
            # The OS hasn't necessarily finished realizing the window's HWND
            # right after the very first imshow() - checking visibility too
            # early can read a not-yet-created window as "already closed"
            # and exit before anything was ever shown, so give it a moment.
            if time.monotonic() - shown_at > 0.3 and cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break  # user closed the window via its title-bar button
    finally:
        cap.release()
        cv2.destroyWindow(title)
    return True
