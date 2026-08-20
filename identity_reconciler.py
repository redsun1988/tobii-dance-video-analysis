import base64
import itertools
from collections import defaultdict

import cv2

from config import AppConfig

app_config = AppConfig()


class PersonIdentityReconciler:
    """
    Pose tracking loses/reassigns track IDs when a dancer is briefly
    occluded or leaves and re-enters frame, fragmenting one real person into
    several IDs and skewing per-person gaze statistics. This collects a few
    sample crops per track ID while the video plays, then after playback
    asks the local VLM to compare crops across ID pairs and merges any pair
    it majority-votes as the same person.

    Only track IDs that received at least one confirmed gaze fixation are
    ever compared - an ID nobody looked at can't skew gaze statistics, so
    reconciling it isn't worth a VLM round-trip. ID pairs that were ever
    visible on screen at the same time are also skipped without querying
    the VLM - two IDs seen simultaneously are provably two different
    physical people, no matter what the VLM says.
    """

    MIN_FRAME_GAP_BETWEEN_SAMPLES = 15  # spreads sampled crops across time instead of clustering them

    def __init__(self):
        self.enabled = app_config.IDENTITY_RECONCILE_ENABLED and app_config.OLLAMA_AVAILABLE
        self._client = None
        self._crops = defaultdict(list)          # person_id -> list of jpeg-encoded crop bytes
        self._last_sampled_frame = {}             # person_id -> frame_idx of last stored sample
        self._first_seen_frame = {}               # person_id -> first frame_idx it was ever detected in
        self._last_seen_frame = {}                # person_id -> last frame_idx it was ever detected in
        self.comparison_log = []                  # one row per pairwise VLM comparison, for CSV export

        if self.enabled:
            from openai import OpenAI
            client_kwargs = {"base_url": app_config.VLM_BASE_URL, "api_key": app_config.VLM_API_KEY}
            self._client = OpenAI(**client_kwargs)
        else:
            print("Person identity reconciliation disabled (Ollama unavailable or IDENTITY_RECONCILE_ENABLED=False).")

    def observe_frame(self, frame_bgr, people_data, frame_idx):
        """Called once per processed frame; tracks each visible person's
        on-screen frame range and opportunistically stores a bounding-box
        crop for it, capped and spaced out over time."""
        if not self.enabled:
            return

        for person in people_data:
            pid = person['id']
            self._first_seen_frame.setdefault(pid, frame_idx)
            self._last_seen_frame[pid] = frame_idx

            crops = self._crops[pid]
            if len(crops) >= app_config.IDENTITY_RECONCILE_CROPS_PER_PERSON:
                continue
            last_sampled = self._last_sampled_frame.get(pid)
            if last_sampled is not None and (frame_idx - last_sampled) < self.MIN_FRAME_GAP_BETWEEN_SAMPLES:
                continue

            crop = self._encode_person_crop(frame_bgr, person['box'])
            if crop is None:
                continue
            crops.append(crop)
            self._last_sampled_frame[pid] = frame_idx

    @staticmethod
    def _encode_person_crop(frame_bgr, box_xyxy):
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box_xyxy
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return None
        ok, buf = cv2.imencode('.jpg', frame_bgr[y1:y2, x1:x2], [int(cv2.IMWRITE_JPEG_QUALITY), app_config.VLM_JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def _ranges_overlap(self, id_a, id_b):
        a0, a1 = self._first_seen_frame[id_a], self._last_seen_frame[id_a]
        b0, b1 = self._first_seen_frame[id_b], self._last_seen_frame[id_b]
        return a0 <= b1 and b0 <= a1

    def estimate_pair_count(self):
        """
        Returns (num_track_ids, num_eligible_pairs): the number of distinct
        track IDs that collected at least one crop, and how many of their
        pairwise combinations are eligible for identity comparison (not
        provably-simultaneous). Lets a caller estimate reconciliation cost
        before running reconcile() - each eligible pair costs at least 2 VLM
        queries (see _same_person's early-exit), and a call can take minutes
        on a CPU-only local VLM, so an exhaustive reconcile(None) pass over
        many fragmented track IDs can be very slow - see video_precomputer.py.
        """
        person_ids = [pid for pid, crops in self._crops.items() if crops]
        all_pairs = list(itertools.combinations(person_ids, 2))
        not_simultaneous = [(a, b) for a, b in all_pairs if not self._ranges_overlap(a, b)]
        return len(person_ids), len(not_simultaneous)

    def reconcile(self, fixated_person_ids):
        """
        Runs after the video ends: compares track IDs pairwise - restricted
        to pairs where both IDs received at least one confirmed gaze
        fixation (see `GazeAnalyzer.fixated_person_ids`), and skipping pairs
        that ever co-existed on screen - and unions any remaining pair the
        VLM majority-votes as the same person. Returns
        {track_id: canonical_id} for every track ID that collected at least
        one crop - IDs that weren't merged (including ones excluded from
        comparison) map to themselves.

        `fixated_person_ids=None` drops the gaze-fixation filter entirely -
        every non-simultaneous pair is compared. Used by the headless
        precompute pipeline (video_precomputer.py), which has no live gaze
        data at all and wants every track ID pair checked for a match, not
        just ones a viewer happened to look at.
        """
        person_ids = [pid for pid, crops in self._crops.items() if crops]
        parent = {pid: pid for pid in person_ids}

        def find(pid):
            while parent[pid] != pid:
                parent[pid] = parent[parent[pid]]
                pid = parent[pid]
            return pid

        if not self.enabled or len(person_ids) < 2:
            return {pid: pid for pid in person_ids}

        all_pairs = list(itertools.combinations(person_ids, 2))
        not_simultaneous = [(a, b) for a, b in all_pairs if not self._ranges_overlap(a, b)]
        if fixated_person_ids is None:
            candidate_pairs = not_simultaneous
            skip_reason = "no gaze-fixation filter applied (exhaustive comparison)"
        else:
            candidate_pairs = [(a, b) for a, b in not_simultaneous if a in fixated_person_ids and b in fixated_person_ids]
            skip_reason = "no confirmed gaze fixation on both IDs"
        print(f"Identity reconciliation: comparing {len(candidate_pairs)} of {len(all_pairs)} track ID pair(s) "
              f"({len(all_pairs) - len(not_simultaneous)} skipped as provably simultaneous people, "
              f"{len(not_simultaneous) - len(candidate_pairs)} skipped - {skip_reason}).")

        for id_a, id_b in candidate_pairs:
            ra, rb = find(id_a), find(id_b)
            if ra == rb:
                continue  # already merged transitively
            if self._same_person(id_a, id_b):
                parent[max(ra, rb)] = min(ra, rb)
                print(f"  Identity match: Person {max(ra, rb)} merged into Person {min(ra, rb)}.")

        return {pid: find(pid) for pid in person_ids}

    def _same_person(self, id_a, id_b):
        """Compares each of id_a's sample crops against id_b's reference
        crop (its first sample) via the VLM, and returns True as soon as a
        majority of "same person" votes is mathematically guaranteed (and
        False as soon as it becomes unreachable), so a clear-cut verdict
        doesn't need to wait for every crop - each query costs ~minutes on
        a CPU-only local VLM."""
        reference = self._crops[id_b][0]
        crops = self._crops[id_a]
        total = len(crops)
        threshold = app_config.IDENTITY_RECONCILE_MAJORITY_THRESHOLD
        yes_count = 0

        for i, crop in enumerate(crops):
            verdict = self._ask_vlm(crop, reference, id_a, id_b, i)
            if verdict is True:
                yes_count += 1

            remaining = total - (i + 1)
            if yes_count > threshold * total:
                return True
            if yes_count + remaining <= threshold * total:
                return False

        return False

    def _ask_vlm(self, crop_bytes, reference_bytes, id_a, id_b, crop_index):
        ref_url = f"data:image/jpeg;base64,{base64.b64encode(reference_bytes).decode('ascii')}"
        crop_url = f"data:image/jpeg;base64,{base64.b64encode(crop_bytes).decode('ascii')}"
        entry = {'person_id_a': id_a, 'person_id_b': id_b, 'crop_index': crop_index}

        try:
            response = self._client.chat.completions.create(
                model=app_config.VLM_MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": app_config.IDENTITY_RECONCILE_PROMPT},
                        {"type": "image_url", "image_url": {"url": ref_url}},
                        {"type": "image_url", "image_url": {"url": crop_url}},
                    ],
                }],
                timeout=app_config.VLM_REQUEST_TIMEOUT_S,
            )
            text = (response.choices[0].message.content or "").strip()
            verdict = text.lower().startswith("yes")
            entry.update({'response': text, 'verdict': verdict, 'error': None})
            self.comparison_log.append(entry)
            return verdict
        except Exception as e:
            entry.update({'response': None, 'verdict': None, 'error': str(e)})
            self.comparison_log.append(entry)
            return None
