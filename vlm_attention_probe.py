import base64
import queue
import threading
import time

import cv2

from config import AppConfig

app_config = AppConfig()


class VlmAttentionProbe:
    """
    Background worker that asks a local vision-capable LLM (via Ollama's
    OpenAI-compatible API) what's visible in a newly-fixated region of the
    frame and what about it might have attracted the viewer's attention.

    Runs on a daemon thread reading off a small bounded queue so a slow (or
    stalled) model call never blocks the capture/analysis loop; when the
    queue backs up, new probes are skipped (and logged as skipped) rather
    than piling up unboundedly.
    """

    def __init__(self):
        self.enabled = app_config.VLM_ENABLED and app_config.OLLAMA_AVAILABLE
        self.query_log = []

        self._client = None
        self._queue = queue.Queue(maxsize=app_config.VLM_QUEUE_MAXSIZE)
        self._log_lock = threading.Lock()
        self._cooldown_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._last_probed_label = None
        self._last_probed_time = None

        if self.enabled:
            from openai import OpenAI
            client_kwargs = {"base_url": app_config.VLM_BASE_URL, "api_key": app_config.VLM_API_KEY}
            self._client = OpenAI(**client_kwargs)
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()
        else:
            print("VLM attention probe disabled (Ollama unavailable or VLM_ENABLED=False).")

    def should_probe(self, target_label, timestamp):
        """
        Cooldown check the caller runs before submit(): a target is worth
        probing when it's a different label than the last one probed, or
        when VLM_COOLDOWN_S has passed since the last probe regardless of
        label. This keeps a genuinely new fixation from being skipped just
        because something else was probed a moment ago, while still
        preventing the same repeated fixation from spamming requests.
        """
        if not self.enabled:
            return False

        with self._cooldown_lock:
            if self._last_probed_label is None:
                return True
            if target_label != self._last_probed_label:
                return True
            if self._last_probed_time is not None and (timestamp - self._last_probed_time) >= app_config.VLM_COOLDOWN_S:
                return True

        return False

    def submit(self, frame_bgr, crop_box_xyxy, target_label, frame_idx, timestamp):
        """
        Non-blocking. Crops `frame_bgr` to `crop_box_xyxy` and queues the
        crop for the worker thread to describe. Safe to call from the main
        capture loop.
        """
        if not self.enabled:
            return

        crop = self._safe_crop(frame_bgr, crop_box_xyxy)
        if crop is None:
            return

        with self._cooldown_lock:
            self._last_probed_label = target_label
            self._last_probed_time = timestamp

        job = {'frame_idx': frame_idx, 'timestamp': timestamp, 'target_label': target_label, 'crop': crop}

        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._append_log({
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'target_label': target_label,
                'status': 'skipped_backlog',
                'latency_s': None,
                'response': None,
                'error': None,
            })

    @staticmethod
    def _safe_crop(frame_bgr, box_xyxy):
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box_xyxy
        x1 = max(0, min(int(round(x1)), w - 1))
        x2 = max(0, min(int(round(x2)), w))
        y1 = max(0, min(int(round(y1)), h - 1))
        y2 = max(0, min(int(round(y2)), h))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2].copy()

    def _append_log(self, entry):
        with self._log_lock:
            self.query_log.append(entry)

    def _worker_loop(self):
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process_job(job)
            finally:
                self._queue.task_done()

    def _process_job(self, job):
        start = time.monotonic()
        try:
            ok, buf = cv2.imencode('.jpg', job['crop'], [int(cv2.IMWRITE_JPEG_QUALITY), app_config.VLM_JPEG_QUALITY])
            if not ok:
                raise RuntimeError("Failed to JPEG-encode crop")

            b64 = base64.b64encode(buf.tobytes()).decode('ascii')
            data_url = f"data:image/jpeg;base64,{b64}"

            response = self._client.chat.completions.create(
                model=app_config.VLM_MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": app_config.VLM_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                timeout=app_config.VLM_REQUEST_TIMEOUT_S,
            )
            text = response.choices[0].message.content
            self._append_log({
                'frame_idx': job['frame_idx'],
                'timestamp': job['timestamp'],
                'target_label': job['target_label'],
                'status': 'ok',
                'latency_s': time.monotonic() - start,
                'response': text,
                'error': None,
            })
        except Exception as e:
            self._append_log({
                'frame_idx': job['frame_idx'],
                'timestamp': job['timestamp'],
                'target_label': job['target_label'],
                'status': 'error',
                'latency_s': time.monotonic() - start,
                'response': None,
                'error': str(e),
            })

    def stop(self, timeout=10.0):
        """Flushes any queued jobs and joins the worker thread, so
        query_log is complete by the time CSV export runs."""
        if not self.enabled or self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
