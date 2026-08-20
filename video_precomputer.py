"""
Headless, no-visible-player precompute of a video's pose/tracking and
demographics cache (see database.py) - decodes the video file directly at a
fixed, denser sample rate (AppConfig.PRECOMPUTE_FPS) and runs identity
reconciliation exhaustively (every eligible track ID pair, not just ones a
live viewer happened to look at), so a later live test session on the same
video can replay this cache instead of recomputing everything from scratch.

Reuses PoseEstimator/PersonIdentityReconciler/PersonDemographicsEstimator -
the same building blocks pose_gaze_application.py uses - but skips
everything gaze-related (EyeGazeTracker, GazeAnalyzer, VlmAttentionProbe)
and the video player window entirely (VideoPlayerWindow), since there's no
live viewer and nothing to display.

Exhaustive identity reconciliation can be very slow: each compared pair
costs at least two local-VLM queries, and a single query can take minutes
on a CPU-only model (see AppConfig.VLM_REQUEST_TIMEOUT_S) - a video with
many fragmented track IDs can realistically take hours to precompute. This
is accepted as a cost of the "high accuracy, run it unattended" mode this
module implements; see PersonIdentityReconciler.estimate_pair_count(),
used here to print a rough order-of-magnitude warning before starting.
"""

import cv2

from config import AppConfig
from database import Database
from pose_estimator import PoseEstimator
from identity_reconciler import PersonIdentityReconciler
from demographics_estimator import PersonDemographicsEstimator

app_config = AppConfig()

# Precompute always decodes the video file directly, so its content rect is
# by construction the whole decoded frame - see video_geometry module
# docstring and pose_estimator.py's content_rect parameter.
PRECOMPUTE_CAPTURE_MODE = 'content'


class VideoPrecomputer:
    """Runs the high-accuracy, no-live-player cache precompute pipeline for
    one or more video files. Always (re)computes fresh and overwrites any
    existing cache for a video, regardless of what's already cached - the
    point of this mode is a deliberate, higher-quality recompute."""

    def __init__(self, db=None):
        self.pose_estimator = PoseEstimator()
        self.identity_reconciler = PersonIdentityReconciler()
        self.demographics_estimator = PersonDemographicsEstimator()
        self.db = db or Database()
        self._owns_db = db is None

    def close(self):
        if self._owns_db:
            self.db.close()

    def precompute(self, video_path):
        """
        Precomputes and caches one video's pose/tracking and (if
        AppConfig.DEMOGRAPHICS_ENABLED) demographics data. Returns a summary
        dict, or None if interrupted before any frame sampling completed
        (nothing was saved in that case). Re-raises KeyboardInterrupt after
        saving whatever phase(s) did complete, so a caller driving a batch
        of videos sees the interrupt and can stop between videos.
        """
        print(f"\nPrecomputing cache for: {video_path}")

        # One capture handle for both metadata and frame sampling - opening
        # a video file/probing its stream info isn't free for every
        # container/codec, so this avoids paying that cost twice per video.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open video file: {video_path}")
        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            video_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
            video_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
            if native_fps <= 0:
                raise RuntimeError(f"Could not determine frame rate of video file: {video_path}")
            if video_w <= 0 or video_h <= 0:
                raise RuntimeError(f"Could not determine native resolution of video file: {video_path}")
            duration_s = frame_count / native_fps if frame_count > 0 else 0.0

            video_id, _ = self.db.get_or_create_video(video_path, duration_s)
            content_rect = (0.0, 0.0, float(video_w), float(video_h))

            pending_rows, sample_count = self._sample_frames(cap, native_fps, content_rect)
        finally:
            cap.release()

        print(f"Pose sampling complete: {sample_count} sample(s) at ~{app_config.PRECOMPUTE_FPS:g} fps "
              f"({len(pending_rows)} person-observation(s) total).")

        id_map, reconciliation_completed, interrupted = self._reconcile(video_id)

        if id_map:
            for row in pending_rows:
                row['track_id'] = id_map.get(row['track_id'], row['track_id'])
        self.db.replace_pose_cache(video_id, pending_rows, PRECOMPUTE_CAPTURE_MODE)
        if reconciliation_completed:
            self.db.set_video_identity_reconciled(video_id, True)
        print(f"Cached pose/tracking data for this video ({sample_count} sample(s)).")

        demographics = {}
        if not interrupted and self.demographics_estimator.enabled:
            try:
                print("Estimating age/gender/build/color for all detected persons...")
                demographics = self.demographics_estimator.estimate(id_map)
                if demographics:
                    self.db.replace_demographics_cache(video_id, demographics)
                    print(f"Cached demographics for {len(demographics)} person(s).")
            except KeyboardInterrupt:
                print("Interrupted during demographics estimation - the pose/reconciliation results "
                      "already computed were still saved.")
                interrupted = True

        if interrupted:
            raise KeyboardInterrupt()

        return {
            'video_id': video_id,
            'sample_count': sample_count,
            'identity_reconciled': reconciliation_completed,
            'demographics_person_count': len(demographics),
        }

    def _sample_frames(self, cap, native_fps, content_rect):
        """
        Reads from the already-open `cap` and feeds one sample every
        `native_fps / AppConfig.PRECOMPUTE_FPS` frames through pose
        estimation, identity-reconciliation crop sampling, and (if enabled)
        demographics crop sampling. `video_time_s` for each sample is
        computed exactly from the frame index and the video's own frame
        rate - no wall-clock jitter, unlike a live capture session.

        Does not release `cap` - the caller (precompute()) owns its
        lifecycle since it also uses it for metadata. If interrupted
        (KeyboardInterrupt), propagates uncaught - the caller then has no
        pending rows to save, so an existing cache for this video is left
        untouched rather than being overwritten with a partial recording.
        """
        step = max(1, round(native_fps / app_config.PRECOMPUTE_FPS))

        pending_rows = []
        sample_idx = 0
        frame_seq = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_seq % step == 0:
                video_time_s = frame_seq / native_fps
                people_data = self.pose_estimator.process_frame(frame)
                cache_rows = PoseEstimator.people_data_to_cache_rows(people_data, content_rect)
                for row in cache_rows:
                    row['sample_seq'] = sample_idx
                    row['video_time_s'] = video_time_s
                pending_rows.extend(cache_rows)

                self.identity_reconciler.observe_frame(frame, people_data, sample_idx)
                self.demographics_estimator.observe_frame(frame, people_data, sample_idx)

                sample_idx += 1
            frame_seq += 1

        return pending_rows, sample_idx

    def _reconcile(self, video_id):
        """
        Runs identity reconciliation exhaustively (every eligible track ID
        pair, not gaze-fixation-gated - see
        PersonIdentityReconciler.reconcile(None)). Returns
        (id_map, reconciliation_completed, interrupted).

        If interrupted mid-reconciliation, no partial merge progress can be
        recovered (reconcile()'s union-find state is local to that call), so
        id_map comes back empty and reconciliation_completed is False - a
        future precompute/live run will retry the whole comparison pass.
        """
        if not self.identity_reconciler.enabled:
            print("Identity reconciliation unavailable (disabled in settings, or the local VLM/Ollama "
                  "server is unreachable) - pose and demographics will still be cached, but fragmented "
                  "track IDs won't be merged this run.")
            return {}, False, False

        num_ids, num_pairs = self.identity_reconciler.estimate_pair_count()
        if num_pairs:
            worst_case_hours = (num_pairs * 2 * app_config.VLM_REQUEST_TIMEOUT_S) / 3600.0
            print(f"Identity reconciliation: {num_ids} distinct track ID(s) seen, {num_pairs} pair(s) to "
                  f"compare exhaustively. Each pair costs at least 2 local-VLM queries (up to "
                  f"{app_config.VLM_REQUEST_TIMEOUT_S:.0f}s each) - up to roughly {worst_case_hours:.1f}h "
                  f"in the worst case. This runs unattended; interrupt with Ctrl+C if needed.")

        try:
            id_map = self.identity_reconciler.reconcile(None)
            return id_map, True, False
        except KeyboardInterrupt:
            print("Interrupted during identity reconciliation - track IDs will stay fragmented for now; "
                  "a future precompute run will retry the merge.")
            return {}, False, True
