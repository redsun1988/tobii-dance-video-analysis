import bisect
import os
import time
from collections import defaultdict

import numpy as np
import win32api
import win32con
from PIL import ImageGrab

from config import AppConfig
from database import Database, CURRENT_POSE_CACHE_FORMAT_VERSION
from gaze_analyzer import GazeAnalyzer
from pose_estimator import PoseEstimator
from eye_gaze_tracker import EyeGazeTracker, GazeUnavailableError
from vlm_attention_probe import VlmAttentionProbe
from identity_reconciler import PersonIdentityReconciler
from demographics_estimator import PersonDemographicsEstimator
from video_window import VideoPlayerWindow, WindowNotFoundError
from video_geometry import read_video_metadata, letterboxed_content_rect
from javelin_thrower import JavelinThrower

app_config = AppConfig()


class _PoseCacheReplay:
    """
    Replays a video's previously-cached pose/tracking data (see
    Database.load_pose_cache) against the current run's actual capture
    timestamps and window size, instead of re-running YOLO-Pose.

    Frame numbers between two runs never line up - frames are screen
    captures paced by wall-clock time, not decoded video frames, so the
    exact number and timing of samples differs every run. Video time
    (seconds since playback started) is the one thing that is comparable
    across runs, so lookups snap to the cached sample whose video_time_s is
    closest to the requested timestamp. Cached coordinates are fractions of
    the video's own content rect, so they scale correctly to the current
    frame's content rect regardless of window position/size/monitor - see
    PoseEstimator.cache_rows_to_people_data and
    video_geometry.letterboxed_content_rect.
    """

    def __init__(self, rows):
        rows_by_seq = defaultdict(list)
        seq_time = {}
        for row in rows:
            rows_by_seq[row['sample_seq']].append(row)
            seq_time[row['sample_seq']] = row['video_time_s']

        ordered_seqs = sorted(seq_time.keys())
        self._rows_by_seq = rows_by_seq
        self._seqs = ordered_seqs
        self._times = [seq_time[seq] for seq in ordered_seqs]

    def __len__(self):
        return len(self._seqs)

    def people_data_at(self, timestamp, content_rect):
        i = bisect.bisect_left(self._times, timestamp)
        if i <= 0:
            nearest = 0
        elif i >= len(self._times):
            nearest = len(self._times) - 1
        else:
            before, after = self._times[i - 1], self._times[i]
            nearest = i - 1 if (timestamp - before) <= (after - timestamp) else i

        rows = self._rows_by_seq[self._seqs[nearest]]
        return PoseEstimator.cache_rows_to_people_data(rows, content_rect)


class PoseGazeApplication:
    """Orchestrates video playback window discovery, gaze tracking, pose
    estimation, gaze-to-body-part analysis, and the VLM attention probe."""

    def __init__(self, viewer_profile=None, db=None):
        self.eye_tracker = EyeGazeTracker()
        # Constructed lazily in run(), only once _setup_video_cache() knows
        # pose actually needs to be computed live rather than replayed from
        # cache - loading the YOLO model is wasted work otherwise (~1-3s,
        # multiplied by every cached video a test playlist replays).
        self.pose_estimator = None
        self.gaze_analyzer = GazeAnalyzer()
        self.vlm_probe = VlmAttentionProbe()
        self.identity_reconciler = PersonIdentityReconciler()
        self.demographics_estimator = PersonDemographicsEstimator()
        # A shared Database (see StartMenu, which owns and closes one for
        # the whole program) avoids reopening a connection and re-running
        # schema setup for every video in a playlist. Falls back to owning
        # a private one - and closing it in _finalize() - for standalone use.
        self.db = db or Database()
        self._owns_db = db is None
        self.frame_idx = 0
        self.viewer_profile = viewer_profile

        # Video analysis cache state - populated at the start of run().
        self.video_id = None
        self._capture_mode = None            # 'content'/'window_uncorrected'/'desktop', see run()
        self._video_native_w = None          # video's own decoded resolution, for content-rect math
        self._video_native_h = None
        self._pose_replay = None             # _PoseCacheReplay; non-None means pose is replayed, not recomputed
        self._pending_pose_frames = []       # buffered cache-row dicts, only populated when NOT replaying
        self._cached_demographics = None     # non-None means demographics are reused from cache
        self._needs_reconciliation_pass = False  # replaying pose, but its IDs were never actually reconciled
        self._completed_normally = True      # False if the loop exited early (window closed / Ctrl+C)
        self._session_start_epoch = None

    @property
    def _pose_cache_hit(self):
        return self._pose_replay is not None

    @property
    def _demographics_cache_hit(self):
        return self._cached_demographics is not None

    def _content_rect_for_frame(self, frame_w, frame_h):
        """Returns (x0, y0, w, h): the sub-rectangle of a `frame_w` x
        `frame_h` captured frame actually occupied by the video's content.
        Only 'content' capture mode (a real player window, with
        AppConfig.PLAYER_PRESERVES_ASPECT_RATIO on) can compute this via
        letterbox-fit math - 'window_uncorrected' and 'desktop' modes fall
        back to treating the whole captured frame as the content rect,
        exactly as before this correction existed."""
        if self._capture_mode == 'content':
            return letterboxed_content_rect(frame_w, frame_h, self._video_native_w, self._video_native_h)
        return (0.0, 0.0, float(frame_w), float(frame_h))

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

        # Everything from here on can raise before the loop even starts
        # (corrupt video file, a DB error) - wrapped in try/finally so a
        # located player window is never left open, and (for a
        # standalone-owned db) the connection isn't leaked, even then.
        # setup_completed also tells the finally block whether _finalize()
        # below will run - if setup itself failed, it won't, so this is the
        # only chance to close an owned db connection.
        setup_completed = False
        try:
            video_duration, self._video_native_w, self._video_native_h = read_video_metadata(video_path)
            if window is not None:
                self._capture_mode = 'content' if app_config.PLAYER_PRESERVES_ASPECT_RATIO else 'window_uncorrected'
            else:
                self._capture_mode = 'desktop'
            self._setup_video_cache(video_path, video_duration)
            if self._pose_replay is None:
                # Cache can't be replayed - pose has to be computed live.
                self.pose_estimator = PoseEstimator()
            setup_completed = True

            start_time = time.time()
            self._session_start_epoch = start_time
            last_time = start_time

            while (time.time() - start_time) < video_duration:
                try:
                    if window is not None and not window.is_alive():
                        print("Video player window closed - stopping early.")
                        self._completed_normally = False
                        break

                    frame, capture_bbox = self._capture_frame(window)
                    frame_h, frame_w = frame.shape[:2]
                    content_rect = self._content_rect_for_frame(frame_w, frame_h)

                    # timestamp is computed here (rather than after pose
                    # estimation, as it originally was) because a cache
                    # replay lookup needs it too, not just gaze duration
                    # accounting.
                    now = time.time()
                    dt = now - last_time
                    last_time = now
                    timestamp = now - start_time

                    if self._pose_replay is not None:
                        people_data = self._pose_replay.people_data_at(timestamp, content_rect)
                        if self._needs_reconciliation_pass:
                            # This cache predates identity reconciliation (or
                            # it was unavailable the first time this video
                            # was analyzed) - sample crops now, from the live
                            # frame, against the replayed (already-canonical-
                            # shaped) boxes, so _finalize() can still run it.
                            self.identity_reconciler.observe_frame(frame, people_data, self.frame_idx)
                    else:
                        people_data = self.pose_estimator.process_frame(frame)
                        cache_rows = PoseEstimator.people_data_to_cache_rows(people_data, content_rect)
                        for cache_row in cache_rows:
                            cache_row['sample_seq'] = self.frame_idx
                            cache_row['video_time_s'] = timestamp
                        self._pending_pose_frames.extend(cache_rows)
                        self.identity_reconciler.observe_frame(frame, people_data, self.frame_idx)

                    if not self._demographics_cache_hit or self._needs_reconciliation_pass:
                        # Also collect crops when a reconciliation pass is
                        # about to run against replayed data, even if
                        # demographics already had a cache hit: a merge
                        # found this run would invalidate that cache (see
                        # _reconcile_identities), and _resolve_demographics()
                        # would otherwise have nothing to recompute from -
                        # see PersonDemographicsEstimator.
                        self.demographics_estimator.observe_frame(frame, people_data, self.frame_idx)

                    try:
                        gaze_x, gaze_y, _ = self.eye_tracker.get_gaze_coords()
                    except GazeUnavailableError as e:
                        print(f"Gaze unavailable this frame - {e}")
                        time.sleep(app_config.GAZE_UNAVAILABLE_RETRY_DELAY_S)
                        self.frame_idx += 1
                        continue

                    gaze_absolute = (gaze_x, gaze_y)
                    gaze_local, in_window = self._to_local(gaze_absolute, capture_bbox)
                    gaze_source = "simulated" if self.eye_tracker.is_simulated else "tobii"

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
                    self._completed_normally = False
                    break
        finally:
            if window is not None:
                window.close()
            if not setup_completed and self._owns_db:
                self.db.close()

        self._finalize()

    def _setup_video_cache(self, video_path, video_duration):
        """
        Looks up this video's fingerprint in the database and decides
        whether pose/tracking and/or demographics can be replayed from a
        previous run's cached analysis instead of recomputed - unless
        AppConfig.FORCE_RECOMPUTE_VIDEO_ANALYSIS says to always recompute
        (the fresh results still overwrite the cache afterwards, so the
        next run benefits again). Must run before the capture loop starts,
        and after self._capture_mode is known.
        """
        self.video_id, _ = self.db.get_or_create_video(video_path, video_duration)
        force_recompute = app_config.FORCE_RECOMPUTE_VIDEO_ANALYSIS

        raw_pose_rows = self.db.load_pose_cache(self.video_id)
        raw_cached_demographics = self.db.load_demographics_cache(self.video_id)
        cache_meta = self.db.get_video_cache_meta(self.video_id)

        pose_rows = [] if force_recompute else raw_pose_rows
        if pose_rows and cache_meta['capture_mode'] != self._capture_mode:
            # The cached fractions are relative to a completely different
            # kind of frame (real player window vs. whole-desktop capture,
            # or corrected vs. uncorrected) - rescaling them against this
            # run's frame would place every cached position somewhere
            # meaningless.
            print(f"Cached pose data for this video was recorded in '{cache_meta['capture_mode']}' capture "
                  f"mode, this run is in '{self._capture_mode}' mode - coordinates wouldn't be comparable, "
                  f"so the cache is being ignored.")
            pose_rows = []
        elif pose_rows and cache_meta['format_version'] != CURRENT_POSE_CACHE_FORMAT_VERSION:
            # The coordinate normalization convention itself changed since
            # this cache was written (see CURRENT_POSE_CACHE_FORMAT_VERSION)
            # - same risk as a capture-mode mismatch, just from a code
            # change instead of a different run environment.
            print(f"Cached pose data for this video was written under an older cache format "
                  f"(v{cache_meta['format_version']}), this build expects v{CURRENT_POSE_CACHE_FORMAT_VERSION} - "
                  f"ignoring the cache.")
            pose_rows = []

        self._pose_replay = _PoseCacheReplay(pose_rows) if pose_rows else None
        self._needs_reconciliation_pass = (
            self._pose_replay is not None
            and not cache_meta['identity_reconciled']
            and self.identity_reconciler.enabled
        )
        self._pending_pose_frames = []

        cached_demographics = {} if force_recompute else raw_cached_demographics
        self._cached_demographics = (
            cached_demographics if (app_config.DEMOGRAPHICS_ENABLED and cached_demographics) else None
        )
        if self._pose_replay is None and self._cached_demographics is not None:
            # Pose is about to be (re)computed fresh, which always assigns
            # new track IDs - any cached demographics reference IDs from a
            # prior analysis pass and can no longer be trusted to line up.
            print("Pose data will be (re)computed fresh for this video - cached demographics reference "
                  "old track IDs, ignoring them too.")
            self._cached_demographics = None

        if self._pose_cache_hit:
            print(f"Video already analyzed before - reusing cached pose/tracking data "
                  f"({len(pose_rows)} cached sample rows) instead of running YOLO-Pose live.")
            if self._needs_reconciliation_pass:
                print("This cache predates identity reconciliation - running it now against the replayed data.")
        if self._demographics_cache_hit:
            print(f"Reusing cached person demographics for {len(cached_demographics)} person(s) "
                  f"from a previous analysis of this video.")
        if force_recompute and (raw_pose_rows or raw_cached_demographics):
            print("FORCE_RECOMPUTE_VIDEO_ANALYSIS is on - ignoring existing cache for this video.")

    def _save_pose_cache(self, id_map):
        """Writes this run's freshly-computed pose/tracking data (already
        lean cache-row dicts, see run()) to the cache, remapped through
        `id_map` (post-hoc identity reconciliation) so cached track IDs are
        already canonical for any future replay."""
        if not self._pending_pose_frames:
            return

        if id_map:
            for row in self._pending_pose_frames:
                row['track_id'] = id_map.get(row['track_id'], row['track_id'])

        self.db.replace_pose_cache(self.video_id, self._pending_pose_frames, self._capture_mode)
        sample_count = len({row['sample_seq'] for row in self._pending_pose_frames})
        print(f"Cached pose/tracking data for this video ({sample_count} frame sample(s)).")

    def _reconcile_identities(self):
        """
        Resolves this run's canonical person id mapping - running identity
        reconciliation live, replaying an already-reconciled cache, or
        running it now against replayed (but never-reconciled) cached pose
        data - and persists the outcome (a fresh pose cache, or an ID remap
        applied to an existing one) so later runs benefit too. Returns the
        {old_id: new_id} map to feed into gaze/demographics.
        """
        if self._pose_cache_hit and not self._needs_reconciliation_pass:
            # Cached track IDs are already canonical (reconciled the first
            # time this video was analyzed, or reconciliation still isn't
            # available/enabled) - nothing to do.
            return {}

        print("Running post-hoc person identity reconciliation...")
        fixated_person_ids = set(self.gaze_analyzer.fixated_person_ids)
        id_map = self.identity_reconciler.reconcile(fixated_person_ids)
        merged_count = sum(1 for old, new in id_map.items() if old != new)

        if merged_count:
            self.gaze_analyzer.remap_person_ids(id_map)
        else:
            print("Identity reconciliation found no matching person pairs to merge.")

        if self._pose_cache_hit:
            # Pose itself was replayed from cache and is still correct -
            # only the ID labels need fixing up, in place. Only commit that
            # to the shared cache from a session that watched the video
            # through to the end: a merge decided from a partial replay
            # (interrupted early) rests on whatever fraction of the track
            # IDs happened to be observed before the interrupt, and once
            # written it can't be reconsidered by a later run (the merged
            # ID no longer exists separately in the cache to re-compare) -
            # so it isn't persisted; a future full playthrough decides fresh.
            if self._completed_normally:
                if merged_count:
                    # Also clears any now-stale demographics cache for this
                    # video (see Database.remap_pose_cache_ids) - no need to
                    # do that here too.
                    self.db.remap_pose_cache_ids(self.video_id, id_map)
                    if self._cached_demographics is not None:
                        print("Identity reconciliation merged person IDs - cached demographics reference "
                              "the old IDs and are now stale, recomputing them fresh.")
                        self._cached_demographics = None
                self.db.set_video_identity_reconciled(self.video_id, True)
            elif merged_count:
                print("Playback stopped early - found matching person pairs, but not committing the merge "
                      "to the shared cache from an incomplete session; a future full playthrough will retry it.")
        elif self._completed_normally:
            self._save_pose_cache(id_map)
        else:
            print("Playback stopped early - not updating the video analysis cache with partial pose data.")

        return id_map

    def _resolve_demographics(self, id_map):
        if self._demographics_cache_hit:
            return self._cached_demographics

        print("Estimating age/gender for all detected persons...")
        demographics = self.demographics_estimator.estimate(id_map)
        if demographics and self._completed_normally:
            self.db.replace_demographics_cache(self.video_id, demographics)
        elif demographics:
            print("Playback stopped early - not caching demographics from a partial session.")
        return demographics

    def _save_session_to_db(self, output_dir, stats):
        user_id = self.db.get_or_create_user(self.viewer_profile)
        session_id = self.db.save_session(
            user_id=user_id,
            video_id=self.video_id,
            output_dir=output_dir,
            pose_cache_used=self._pose_cache_hit,
            demographics_cache_used=self._demographics_cache_hit,
            stats=stats,
            started_at_epoch=self._session_start_epoch,
        )
        self.db.save_session_person_gaze(session_id, stats['person_gaze_duration'], stats['person_gaze_ratio'])
        self.db.save_session_body_part_gaze(session_id, stats['part_gaze_duration'], stats['part_gaze_ratio'])
        print(f"Session results saved to local database (session id {session_id}).")

    def _finalize(self):
        try:
            print("Waiting for pending VLM attention-probe queries to finish...")
            self.vlm_probe.stop()

            id_map = self._reconcile_identities()
            demographics = self._resolve_demographics(id_map)

            stats = self.gaze_analyzer.generate_statistics(demographics)
            self.gaze_analyzer.display_statistics(stats, demographics=demographics)

            run_id = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(app_config.CSV_OUTPUT_DIR, run_id)
            os.makedirs(output_dir, exist_ok=True)

            extra_summary_fields = {
                'pose_cache_used': self._pose_cache_hit,
                'demographics_cache_used': self._demographics_cache_hit,
            }
            try:
                self.gaze_analyzer.save_to_csv(
                    output_dir,
                    vlm_query_log=self.vlm_probe.query_log,
                    identity_comparison_log=self.identity_reconciler.comparison_log,
                    demographics=demographics,
                    demographics_query_log=self.demographics_estimator.query_log,
                    stats=stats,
                    viewer_profile=self.viewer_profile,
                    extra_summary_fields=extra_summary_fields,
                )
                self.gaze_analyzer.save_to_excel(
                    os.path.join(output_dir, "gaze_statistics.xlsx"),
                    vlm_query_log=self.vlm_probe.query_log,
                    identity_comparison_log=self.identity_reconciler.comparison_log,
                    demographics=demographics,
                    demographics_query_log=self.demographics_estimator.query_log,
                    stats=stats,
                    viewer_profile=self.viewer_profile,
                    extra_summary_fields=extra_summary_fields,
                )
            except Exception as e:
                print(f"Warning: failed to write CSV/Excel output files: {e}")

            try:
                self._save_session_to_db(output_dir, stats)
            except Exception as e:
                print(f"Warning: failed to save session results to the database: {e}")
        finally:
            if self._owns_db:
                self.db.close()
