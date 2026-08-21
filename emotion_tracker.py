"""Background webcam capture + facial emotion recognition: records what
emotion the tested viewer's face is showing over the course of a session,
timestamped on the same video-relative time axis gaze events use (seconds
since playback started - see PoseGazeApplication._session_start_epoch), so
a later analysis can correlate "what was on screen / where the gaze was"
with "how the viewer's face reacted to it".

Uses DeepFace's bundled facial-expression classifier (see
AppConfig.EMOTION_DETECTOR_BACKEND) instead of the local VLM used elsewhere
in this app (vlm_attention_probe.py/demographics_estimator.py) - a
dedicated model trained specifically for facial expression recognition is
both far faster (tens of milliseconds vs. minutes per query) and more
reliable at this narrow task than a general-purpose vision-language model
prompted to guess an emotion word.

Runs on its own daemon thread, opening the configured webcam
(AppConfig.WEBCAM_INDEX - see webcam.py) and polling it at
AppConfig.EMOTION_SAMPLE_INTERVAL_S intervals, independent of the main
capture/gaze loop's own pace. A missing/unselected webcam, disabled
tracking, or a failed camera open are all soft failures (logged, tracking
simply produces no data) rather than aborting the session - unlike the
Tobii gaze tracker, this is a supplementary analytic, not the gaze-tracking
test itself, so there's no case where silently producing less data here
could be mistaken for a real (but fabricated) result.
"""

import threading
import time

from config import AppConfig

app_config = AppConfig()


class EmotionTracker:
    """Owns one webcam capture + background emotion-analysis thread for one
    PoseGazeApplication run. Construct fresh per video, like the other
    per-run analyzers (PersonIdentityReconciler, PersonDemographicsEstimator)."""

    def __init__(self):
        self.enabled = (
            app_config.EMOTION_TRACKING_ENABLED
            and app_config.DEEPFACE_AVAILABLE
            and app_config.WEBCAM_INDEX is not None
        )
        self.records = []  # one dict per successful poll, the raw timeline for CSV export
        self._records_lock = threading.Lock()
        self._cap = None
        self._analyze = None
        self._stop_event = threading.Event()
        self._thread = None
        self._start_epoch = None
        self._sample_seq = 0

        if not self.enabled:
            reason = (
                "disabled in settings" if not app_config.EMOTION_TRACKING_ENABLED else
                "deepface not installed" if not app_config.DEEPFACE_AVAILABLE else
                "no webcam selected (see start menu -> Settings)"
            )
            print(f"Webcam emotion tracking disabled ({reason}).")

    def start(self, session_start_epoch):
        """Opens the configured webcam and starts the background polling
        thread. `session_start_epoch` must be the same time.time() reference
        the caller uses for its own gaze-event timestamps, so emotion sample
        timestamps land on the same axis and can be correlated later."""
        if not self.enabled:
            return

        import cv2
        self._cap = cv2.VideoCapture(app_config.WEBCAM_INDEX, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            print(f"Warning: could not open webcam #{app_config.WEBCAM_INDEX} - "
                  "emotion tracking disabled for this session.")
            self._cap.release()
            self._cap = None
            self.enabled = False
            return

        # Imported lazily (like the OpenAI client in vlm_attention_probe.py) so
        # a process that never actually uses emotion tracking doesn't pay
        # DeepFace/TensorFlow's import cost for nothing.
        from deepface import DeepFace
        self._analyze = DeepFace.analyze

        self._start_epoch = session_start_epoch
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        print(f"Webcam emotion tracking started (camera #{app_config.WEBCAM_INDEX}).")

    def _worker_loop(self):
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            ok, frame = self._cap.read()
            timestamp = time.time() - self._start_epoch
            if ok and frame is not None:
                self._process_frame(frame, timestamp)
            elapsed = time.monotonic() - loop_start
            self._stop_event.wait(max(0.0, app_config.EMOTION_SAMPLE_INTERVAL_S - elapsed))

    def _process_frame(self, frame_bgr, timestamp):
        sample_seq = self._sample_seq
        self._sample_seq += 1
        try:
            results = self._analyze(
                frame_bgr,
                actions=['emotion'],
                detector_backend=app_config.EMOTION_DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True,
            )
            # enforce_detection=False means a frame with no face still comes
            # back as one result (covering the whole frame) with
            # face_confidence == 0, rather than raising - picking the
            # highest-confidence face handles both "no face" and the rare
            # case of more than one face in frame (only the primary test
            # subject should be in view of this webcam, but nothing stops a
            # bystander from wandering into frame).
            face = max(results, key=lambda r: r.get('face_confidence', 0)) if results else None
            face_detected = bool(face) and face.get('face_confidence', 0) > 0
            entry = {
                'sample_seq': sample_seq,
                'timestamp': timestamp,
                'face_detected': face_detected,
                'dominant_emotion': face['dominant_emotion'] if face_detected else None,
                'emotion_scores': {k: float(v) for k, v in face['emotion'].items()} if face_detected else {},
                'error': None,
            }
        except Exception as e:
            entry = {
                'sample_seq': sample_seq,
                'timestamp': timestamp,
                'face_detected': False,
                'dominant_emotion': None,
                'emotion_scores': {},
                'error': str(e),
            }

        with self._records_lock:
            self.records.append(entry)

    def stop(self, timeout=5.0):
        """Stops the polling thread and releases the webcam. Safe to call
        even if start() was never called or tracking is disabled."""
        if not self.enabled or self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print(f"Webcam emotion tracking stopped ({len(self.records)} sample(s) recorded).")

    # --- Statistics / export ---------------------------------------------------

    def generate_statistics(self):
        """
        Aggregates the recorded timeline into per-emotion durations, a
        dominant overall emotion, an emotion-change timeline (analogous to
        GazeAnalyzer's gaze transitions), and the fraction of samples where a
        face was actually found. Each sample is treated as representing the
        interval up to the next sample (or, for the last sample, a
        zero-length interval) - the same dt-based duration accounting
        GazeAnalyzer uses for gaze targets.
        """
        empty = {
            'sample_count': 0, 'face_detected_ratio': 0.0, 'emotion_duration': {},
            'emotion_ratio': {}, 'dominant_emotion': None, 'emotion_transitions': [],
        }
        with self._records_lock:
            records = sorted(self.records, key=lambda r: r['timestamp'])
        if not records:
            return empty

        duration = {label: 0.0 for label in app_config.EMOTION_LABELS}
        transitions = []
        last_emotion = None
        detected_count = 0

        for i, record in enumerate(records):
            next_ts = records[i + 1]['timestamp'] if i + 1 < len(records) else record['timestamp']
            dt = max(0.0, next_ts - record['timestamp'])
            if record['face_detected']:
                detected_count += 1
                emotion = record['dominant_emotion']
                duration[emotion] = duration.get(emotion, 0.0) + dt
                if last_emotion is not None and emotion != last_emotion:
                    transitions.append({'timestamp': record['timestamp'], 'from': last_emotion, 'to': emotion})
                last_emotion = emotion

        total = sum(duration.values())
        ratio = {label: (d / total if total > 0 else 0.0) for label, d in duration.items()}
        dominant = max(duration, key=duration.get) if total > 0 else None

        return {
            'sample_count': len(records),
            'face_detected_ratio': detected_count / len(records),
            'emotion_duration': duration,
            'emotion_ratio': ratio,
            'dominant_emotion': dominant,
            'emotion_transitions': transitions,
        }

    def display_statistics(self, stats):
        """Prints the calculated emotion statistics to the console."""
        print("\n--- Webcam Emotion Statistics ---")
        if stats['sample_count'] == 0:
            print("No webcam emotion samples recorded.")
            return

        print(f"Samples: {stats['sample_count']}, face detected in {stats['face_detected_ratio']:.1%} of them")
        print("Emotion duration:")
        for label, d in sorted(stats['emotion_duration'].items(), key=lambda kv: kv[1], reverse=True):
            if d > 0:
                print(f"  {label}: {d:.2f}s ({stats['emotion_ratio'][label]:.2%})")
        print(f"Dominant emotion: {stats['dominant_emotion'] or 'N/A'}")
        print(f"Emotion changes: {len(stats['emotion_transitions'])}")

    def get_records(self):
        """Thread-safe copy of the raw per-sample timeline (sample_seq,
        timestamp, face_detected, dominant_emotion, emotion_scores, error).
        `timestamp` is video-relative (seconds since playback started - the
        same axis PoseGazeApplication stamps pose cache rows' video_time_s
        with), so callers can persist it as such - see
        Database.save_session_emotion_samples. Unlike get_export_rows(),
        this keeps emotion_scores as a dict instead of splitting it into
        separate per-emotion columns."""
        with self._records_lock:
            return list(self.records)

    def get_export_rows(self):
        """Flattens the raw per-sample timeline into rows suitable for a
        pandas.DataFrame - one row per webcam poll, one column per emotion
        label's score - for the gaze_analyzer.py CSV/Excel timeline export."""
        with self._records_lock:
            records = list(self.records)

        rows = []
        for r in records:
            row = {
                'sample_seq': r['sample_seq'],
                'timestamp': r['timestamp'],
                'face_detected': r['face_detected'],
                'dominant_emotion': r['dominant_emotion'],
                'error': r['error'],
            }
            for label in app_config.EMOTION_LABELS:
                row[f'{label}_score'] = r['emotion_scores'].get(label)
            rows.append(row)
        return rows
