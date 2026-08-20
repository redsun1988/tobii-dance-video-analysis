"""
Local SQLite database for this app: viewer ("user") profiles, per-session
test results, and a per-video cache of the expensive pose/tracking and
VLM-demographics analysis.

Analyzing a video is costly - YOLO-Pose runs every captured frame, and the
local VLM demographics/identity queries take on the order of minutes each
(see AppConfig.VLM_REQUEST_TIMEOUT_S). Since the same video is typically
shown to many test subjects, caching that video's analysis once and reusing
it across every later session (any viewer, any run) is a large speedup. See
AppConfig.FORCE_RECOMPUTE_VIDEO_ANALYSIS to bypass the cache deliberately.

Cached body positions are stored as fractions of the video's own content
rect within the recording-time capture frame (not the raw frame, and not
absolute pixels) - see PoseEstimator.people_data_to_cache_rows/
cache_rows_to_people_data and video_geometry.letterboxed_content_rect. That
keeps them valid when the video player window is later moved, resized, or
shown on a different monitor, and lets a live window-capture session and a
headless precompute (video_precomputer.py, which decodes the file directly)
share the exact same coordinate space and reuse each other's cache.

Cached positions are only comparable, however, within the same *capture
mode* (a real player window's content rect vs. a whole-desktop fallback
capture with no content-rect correction possible) and the same *cache
format version* (bumped whenever the coordinate normalization convention
itself changes, so an older cache written under a since-changed convention
is never silently misinterpreted) - see `pose_cache_capture_mode` and
`pose_cache_format_version` below, and CURRENT_POSE_CACHE_FORMAT_VERSION.
"""

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

from config import AppConfig
from viewer_profile import PROFILE_FIELDS

app_config = AppConfig()

# Bumped whenever PoseEstimator.people_data_to_cache_rows/
# cache_rows_to_people_data's coordinate normalization convention changes,
# so a cache written under an earlier convention is never replayed under a
# newer one it's not actually compatible with (see get_video_cache_meta).
# v2: coordinates normalized against the video's letterbox-corrected
# content rect (video_geometry.letterboxed_content_rect) instead of the
# raw captured frame.
CURRENT_POSE_CACHE_FORMAT_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    gender TEXT,
    age INTEGER,
    occupation TEXT,
    dance_experience TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL UNIQUE,
    file_path TEXT,
    duration_s REAL,
    created_at TEXT NOT NULL,
    -- 'content' (real player window, content-rect-corrected - or a headless
    -- precompute, which is equivalent by construction), 'window_uncorrected'
    -- (real player window, AppConfig.PLAYER_PRESERVES_ASPECT_RATIO was off),
    -- or 'desktop' (whole-screen fallback capture, no content rect knowable
    -- at all) - cached pose fractions from one mode are not comparable to a
    -- run in another, since the captured frame means something different
    -- (see PoseGazeApplication._setup_video_cache).
    pose_cache_capture_mode TEXT,
    -- See CURRENT_POSE_CACHE_FORMAT_VERSION.
    pose_cache_format_version INTEGER,
    -- Whether the cached pose data's track IDs have actually been through a
    -- successful identity-reconciliation pass (as opposed to just having
    -- never been fragmented in the first place, or reconciliation having
    -- been disabled/unavailable the one time this video was analyzed).
    identity_reconciled INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS video_pose_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    sample_seq INTEGER NOT NULL,
    video_time_s REAL NOT NULL,
    track_id INTEGER NOT NULL,
    box_json TEXT NOT NULL,
    body_parts_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pose_cache_video_seq ON video_pose_cache(video_id, sample_seq);

CREATE TABLE IF NOT EXISTS video_person_demographics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,
    age TEXT,
    gender TEXT,
    body_build TEXT,
    dominant_color TEXT,
    sample_count INTEGER,
    age_votes_json TEXT,
    gender_votes_json TEXT,
    body_build_votes_json TEXT,
    dominant_color_votes_json TEXT,
    UNIQUE(video_id, track_id)
);

CREATE TABLE IF NOT EXISTS test_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    video_id INTEGER NOT NULL REFERENCES videos(id),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    output_dir TEXT,
    pose_cache_used INTEGER NOT NULL,
    demographics_cache_used INTEGER NOT NULL,
    total_frames INTEGER,
    total_time_s REAL,
    total_gaze_time_s REAL,
    saccade_count INTEGER,
    confirmed_fixation_count INTEGER,
    most_gazed_person INTEGER,
    most_gazed_part TEXT
);

CREATE TABLE IF NOT EXISTS session_person_gaze (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES test_sessions(id) ON DELETE CASCADE,
    person_track_id INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    ratio REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS session_body_part_gaze (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES test_sessions(id) ON DELETE CASCADE,
    body_part TEXT NOT NULL,
    duration_s REAL NOT NULL,
    ratio REAL NOT NULL
);
"""

# (table, column, type declaration) columns that may be missing on a
# database file created by an earlier version of this schema. Applied via
# ALTER TABLE after CREATE TABLE IF NOT EXISTS, which never adds columns to
# an already-existing table on its own.
_MIGRATIONS = (
    ("videos", "pose_cache_capture_mode", "TEXT"),
    ("videos", "identity_reconciled", "INTEGER NOT NULL DEFAULT 0"),
    ("test_sessions", "finished_at", "TEXT"),
    ("videos", "pose_cache_format_version", "INTEGER"),
)


class Database:
    """One SQLite connection per instance - construct once per process (see
    StartMenu) and share it, rather than one per video/query: each instance
    re-creates the schema and is otherwise cheap, but a fresh sqlite3
    connection still isn't free to open repeatedly in a loop."""

    def __init__(self, db_path=None):
        self.db_path = db_path or app_config.DB_PATH
        parent_dir = os.path.dirname(self.db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    def _migrate(self):
        for table, column, decl in _MIGRATIONS:
            existing = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()

    @staticmethod
    def _now(epoch_seconds=None):
        """ISO-8601 UTC timestamp for `epoch_seconds` (a time.time() value),
        or for the current moment if omitted."""
        if epoch_seconds is None:
            return datetime.now(timezone.utc).isoformat()
        return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()

    # --- Video identity -----------------------------------------------------

    @staticmethod
    def compute_file_hash(path, sample_size=4 * 1024 * 1024):
        """
        A fast fingerprint (file size + hash of its first/last few MB)
        rather than a full-file hash, so identifying an already-analyzed
        video doesn't mean re-reading a potentially huge file end to end on
        every run. Collisions would require two distinct videos sharing the
        exact same size and first/last bytes, which isn't a realistic
        concern here.
        """
        size = os.path.getsize(path)
        hasher = hashlib.sha256()
        hasher.update(str(size).encode("utf-8"))
        with open(path, "rb") as f:
            hasher.update(f.read(sample_size))
            if size > sample_size:
                f.seek(max(size - sample_size, 0))
                hasher.update(f.read(sample_size))
        return hasher.hexdigest()

    def get_or_create_video(self, file_path, duration_s):
        """Returns (video_id, file_hash) for this video file, creating a new
        `videos` row on first sight of this fingerprint. Refreshes the
        stored path/duration on repeat sightings (the file may have moved)."""
        file_hash = self.compute_file_hash(file_path)
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM videos WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            if row:
                video_id = row["id"]
                self._conn.execute(
                    "UPDATE videos SET file_path = ?, duration_s = ? WHERE id = ?",
                    (file_path, duration_s, video_id),
                )
                self._conn.commit()
                return video_id, file_hash

            cur = self._conn.execute(
                "INSERT INTO videos (file_hash, file_path, duration_s, created_at) VALUES (?, ?, ?, ?)",
                (file_hash, file_path, duration_s, self._now()),
            )
            self._conn.commit()
            return cur.lastrowid, file_hash

    def get_video_cache_meta(self, video_id):
        """Returns {'capture_mode': str|None, 'format_version': int|None,
        'identity_reconciled': bool} for this video's cached analysis - used
        to decide whether an existing pose cache is actually safe to replay
        (same capture mode AND same format version - see
        CURRENT_POSE_CACHE_FORMAT_VERSION) and whether its track IDs still
        need an identity-reconciliation pass."""
        with self._lock:
            row = self._conn.execute(
                "SELECT pose_cache_capture_mode, pose_cache_format_version, identity_reconciled "
                "FROM videos WHERE id = ?",
                (video_id,),
            ).fetchone()
        if row is None:
            return {"capture_mode": None, "format_version": None, "identity_reconciled": False}
        return {
            "capture_mode": row["pose_cache_capture_mode"],
            "format_version": row["pose_cache_format_version"],
            "identity_reconciled": bool(row["identity_reconciled"]),
        }

    def set_video_identity_reconciled(self, video_id, value):
        with self._lock:
            self._conn.execute(
                "UPDATE videos SET identity_reconciled = ? WHERE id = ?", (int(value), video_id)
            )
            self._conn.commit()

    # --- Pose/tracking cache -------------------------------------------------

    def load_pose_cache(self, video_id):
        """Returns cached pose samples ordered by capture sequence, as a
        list of {sample_seq, video_time_s, track_id, box, body_parts} - box
        and body_parts endpoints are fractions of the recording-time video
        content rect (see PoseEstimator.cache_rows_to_people_data). Callers
        should check get_video_cache_meta() first - these fractions are only
        meaningful together with a matching capture_mode/format_version."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT sample_seq, video_time_s, track_id, box_json, body_parts_json "
                "FROM video_pose_cache WHERE video_id = ? ORDER BY sample_seq",
                (video_id,),
            ).fetchall()
        return [
            {
                "sample_seq": r["sample_seq"],
                "video_time_s": r["video_time_s"],
                "track_id": r["track_id"],
                "box": json.loads(r["box_json"]),
                "body_parts": json.loads(r["body_parts_json"]),
            }
            for r in rows
        ]

    def replace_pose_cache(self, video_id, rows, capture_mode):
        """Overwrites this video's cached pose data with `rows` (each a
        {sample_seq, video_time_s, track_id, box, body_parts} dict, as
        produced by PoseEstimator.people_data_to_cache_rows) and records the
        capture mode ('content', 'window_uncorrected' or 'desktop' - see the
        `videos` table definition) it was recorded under, stamped with
        CURRENT_POSE_CACHE_FORMAT_VERSION, so a later run in a different
        capture mode or under an older format version knows not to replay it
        (see get_video_cache_meta). Freshly (re)computed pose data means any
        existing demographics cache's track IDs are no longer trustworthy,
        so it's cleared here too - callers that already recomputed fresh
        demographics this run should re-save them afterwards."""
        with self._lock:
            self._conn.execute("DELETE FROM video_pose_cache WHERE video_id = ?", (video_id,))
            self._conn.executemany(
                "INSERT INTO video_pose_cache "
                "(video_id, sample_seq, video_time_s, track_id, box_json, body_parts_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        video_id, r["sample_seq"], r["video_time_s"], r["track_id"],
                        json.dumps(r["box"]), json.dumps(r["body_parts"]),
                    )
                    for r in rows
                ],
            )
            self._conn.execute(
                "UPDATE videos SET pose_cache_capture_mode = ?, pose_cache_format_version = ?, "
                "identity_reconciled = 0 WHERE id = ?",
                (capture_mode, CURRENT_POSE_CACHE_FORMAT_VERSION, video_id),
            )
            self._conn.execute("DELETE FROM video_person_demographics WHERE video_id = ?", (video_id,))
            self._conn.commit()

    def remap_pose_cache_ids(self, video_id, id_map):
        """Rewrites cached pose rows' track_id in place through `id_map`
        ({old_id: new_id}), for merges found by identity reconciliation
        running against replayed (already cached) pose data - see
        PoseGazeApplication._reconcile_identities. Only rewrites entries
        that actually change; a no-op if id_map has no real merges.

        Also clears this video's demographics cache whenever a real merge
        happens, unconditionally - any cached demographics row is keyed by
        a track_id that may no longer exist under that id after the remap,
        so it can no longer be trusted to line up. This is done here rather
        than left to the caller so it can't be forgotten or skipped based
        on unrelated in-memory state (e.g. whether *this* run happened to
        have demographics caching enabled) - see replace_pose_cache, which
        does the same unconditionally for a full pose recompute."""
        changes = [(new, old) for old, new in id_map.items() if old != new]
        if not changes:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE video_pose_cache SET track_id = ? WHERE video_id = ? AND track_id = ?",
                [(new, video_id, old) for new, old in changes],
            )
            self._conn.execute("DELETE FROM video_person_demographics WHERE video_id = ?", (video_id,))
            self._conn.commit()

    # --- Demographics cache ---------------------------------------------------

    def load_demographics_cache(self, video_id):
        """Returns {track_id: {age, gender, body_build, dominant_color,
        sample_count, *_votes}} for every person previously judged for this
        video, or {} if none cached yet."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT track_id, age, gender, body_build, dominant_color, sample_count, "
                "age_votes_json, gender_votes_json, body_build_votes_json, dominant_color_votes_json "
                "FROM video_person_demographics WHERE video_id = ?",
                (video_id,),
            ).fetchall()
        return {
            r["track_id"]: {
                "age": r["age"],
                "gender": r["gender"],
                "body_build": r["body_build"],
                "dominant_color": r["dominant_color"],
                "sample_count": r["sample_count"],
                "age_votes": json.loads(r["age_votes_json"]),
                "gender_votes": json.loads(r["gender_votes_json"]),
                "body_build_votes": json.loads(r["body_build_votes_json"]),
                "dominant_color_votes": json.loads(r["dominant_color_votes_json"]),
            }
            for r in rows
        }

    def replace_demographics_cache(self, video_id, results):
        """Overwrites this video's cached demographics with `results`
        (PersonDemographicsEstimator.estimate() output, keyed by canonical
        person id)."""
        with self._lock:
            self._conn.execute("DELETE FROM video_person_demographics WHERE video_id = ?", (video_id,))
            self._conn.executemany(
                "INSERT INTO video_person_demographics "
                "(video_id, track_id, age, gender, body_build, dominant_color, sample_count, "
                "age_votes_json, gender_votes_json, body_build_votes_json, dominant_color_votes_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        video_id, pid, r.get("age"), r.get("gender"), r.get("body_build"),
                        r.get("dominant_color"), r.get("sample_count"),
                        json.dumps(r.get("age_votes", {})), json.dumps(r.get("gender_votes", {})),
                        json.dumps(r.get("body_build_votes", {})), json.dumps(r.get("dominant_color_votes", {})),
                    )
                    for pid, r in results.items()
                ],
            )
            self._conn.commit()

    # --- Viewers / results -----------------------------------------------------

    def list_users(self):
        """Returns every viewer profile stored so far, most recently created
        first, as a list of {id, full_name, gender, age, occupation,
        dance_experience, created_at} dicts - used to let a new session pick
        an already-known viewer instead of retyping their profile."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, full_name, gender, age, occupation, dance_experience, created_at "
                "FROM users ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_or_create_user(self, profile):
        """Returns a user id for this viewer profile dict (see
        viewer_profile.PROFILE_FIELDS), reusing an existing row on an exact
        repeat of the same profile. Returns None if no profile was collected
        for this session."""
        if not profile or not profile.get("full_name"):
            return None

        values = tuple(profile.get(f) for f in PROFILE_FIELDS)
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM users WHERE full_name = ? AND gender IS ? AND age IS ? "
                "AND occupation IS ? AND dance_experience IS ?",
                values,
            ).fetchone()
            if row:
                return row["id"]

            cur = self._conn.execute(
                "INSERT INTO users (full_name, gender, age, occupation, dance_experience, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                values + (self._now(),),
            )
            self._conn.commit()
            return cur.lastrowid

    def save_session(self, user_id, video_id, output_dir, pose_cache_used, demographics_cache_used,
                      stats, started_at_epoch):
        """Records one test session's summary. `started_at_epoch` is the
        time.time() value captured when playback actually began (not when
        this method runs, which is only after the full video played and
        post-hoc VLM analysis finished). Returns the new session id."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO test_sessions (user_id, video_id, started_at, finished_at, output_dir, "
                "pose_cache_used, demographics_cache_used, total_frames, total_time_s, total_gaze_time_s, "
                "saccade_count, confirmed_fixation_count, most_gazed_person, most_gazed_part) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id, video_id, self._now(started_at_epoch), self._now(), output_dir,
                    int(pose_cache_used), int(demographics_cache_used),
                    stats.get("total_frames"), stats.get("total_time"), stats.get("total_gaze_time"),
                    stats.get("saccade_count"), stats.get("confirmed_fixation_count"),
                    stats.get("most_gazed_person"), stats.get("most_gazed_part"),
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def save_session_person_gaze(self, session_id, duration_by_person, ratio_by_person):
        if not duration_by_person:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO session_person_gaze (session_id, person_track_id, duration_s, ratio) "
                "VALUES (?, ?, ?, ?)",
                [
                    (session_id, pid, duration, ratio_by_person.get(pid, 0.0))
                    for pid, duration in duration_by_person.items()
                ],
            )
            self._conn.commit()

    def save_session_body_part_gaze(self, session_id, duration_by_part, ratio_by_part):
        if not duration_by_part:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO session_body_part_gaze (session_id, body_part, duration_s, ratio) "
                "VALUES (?, ?, ?, ?)",
                [
                    (session_id, part, duration, ratio_by_part.get(part, 0.0))
                    for part, duration in duration_by_part.items()
                ],
            )
            self._conn.commit()
