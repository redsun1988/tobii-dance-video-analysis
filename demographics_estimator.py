import base64
import re
from collections import Counter, defaultdict
from itertools import zip_longest

import cv2

from config import AppConfig

app_config = AppConfig()


class PersonDemographicsEstimator:
    """
    Estimates a small set of appearance attributes - age category, gender,
    body build, and dominant clothing color - for every detected person. A
    single frame is a noisy source (motion blur, a bad angle, partial
    occlusion), so - like PersonIdentityReconciler - this samples a few
    crops per track ID while the video plays, then after playback asks the
    local VLM to judge each sampled crop independently and takes a majority
    vote per attribute across those per-frame judgments. An attribute is
    only reported once its winning category clears
    AppConfig.DEMOGRAPHICS_MAJORITY_THRESHOLD of that attribute's non-null
    votes; otherwise it's left unresolved (None) rather than reporting a
    bare plurality from a near-tied vote.
    """

    MIN_FRAME_GAP_BETWEEN_SAMPLES = 15  # spreads sampled crops across time instead of clustering them

    # attribute name -> AppConfig attribute holding its allowed category values.
    # Order here also controls the order the VLM is asked to answer in (see
    # AppConfig.DEMOGRAPHICS_PROMPT) and the order values are parsed back out.
    ATTRIBUTE_CATEGORIES_CONFIG = {
        'age': 'DEMOGRAPHICS_AGE_CATEGORIES',
        'gender': 'DEMOGRAPHICS_GENDERS',
        'body_build': 'DEMOGRAPHICS_BODY_BUILDS',
        'dominant_color': 'DEMOGRAPHICS_CLOTHING_COLORS',
    }

    def __init__(self):
        self.enabled = app_config.DEMOGRAPHICS_ENABLED and app_config.OLLAMA_AVAILABLE
        self._client = None
        self._crops = defaultdict(list)          # person_id -> list of jpeg-encoded crop bytes
        self._last_sampled_frame = {}             # person_id -> frame_idx of last stored sample
        self._categories = {                      # attribute -> tuple of allowed category values
            attr: getattr(app_config, cfg_name) for attr, cfg_name in self.ATTRIBUTE_CATEGORIES_CONFIG.items()
        }
        self.query_log = []                       # one row per per-crop VLM judgment, for CSV export
        self.results = {}                          # canonical person_id -> {attr, attr_votes for each attribute, 'sample_count'}

        if self.enabled:
            from openai import OpenAI
            client_kwargs = {"base_url": app_config.VLM_BASE_URL, "api_key": app_config.VLM_API_KEY}
            self._client = OpenAI(**client_kwargs)
        else:
            print("Person demographics estimation disabled (Ollama unavailable or DEMOGRAPHICS_ENABLED=False).")

    def observe_frame(self, frame_bgr, people_data, frame_idx):
        """Called once per processed frame; opportunistically stores a
        bounding-box crop for each visible person, capped and spaced out
        over time, exactly like PersonIdentityReconciler.observe_frame."""
        if not self.enabled:
            return

        for person in people_data:
            pid = person['id']
            crops = self._crops[pid]
            if len(crops) >= app_config.DEMOGRAPHICS_CROPS_PER_PERSON:
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

    def estimate(self, id_map=None):
        """
        Runs after the video ends (and after identity reconciliation, if
        used): pools each track ID's sampled crops onto its canonical person
        ID via id_map, then - for every canonical ID that collected at least
        one crop - judges each sampled crop independently via the VLM and
        majority-votes across those judgments, per attribute.

        Arguments:
            id_map: optional {track_id: canonical_id} from
                PersonIdentityReconciler.reconcile(), used to pool crops of
                merged track IDs before judging.

        Returns {person_id: {'age', 'gender', 'body_build', 'dominant_color',
        'age_votes', 'gender_votes', 'body_build_votes',
        'dominant_color_votes', 'sample_count'}}.
        """
        self.results = {}
        if not self.enabled:
            return self.results

        id_map = id_map or {}

        # Group crops by canonical person, keeping each source track ID's
        # crops as its own list (not yet flattened) - track IDs with no
        # successfully-encoded crop are dropped here so they don't produce a
        # phantom all-None result later.
        crop_lists_by_person = defaultdict(list)
        for pid, crops in self._crops.items():
            if crops:
                crop_lists_by_person[id_map.get(pid, pid)].append(crops)

        # Interleave crops from each merged track ID (round-robin) before
        # re-applying the per-person cap, so merging several fragmented
        # track IDs into one canonical person can't multiply their crop
        # count past DEMOGRAPHICS_CROPS_PER_PERSON, while still keeping a
        # mix from every merged fragment rather than only the first one.
        pooled_crops = {}
        for pid, track_crop_lists in crop_lists_by_person.items():
            interleaved = [c for group in zip_longest(*track_crop_lists) for c in group if c is not None]
            pooled_crops[pid] = interleaved[:app_config.DEMOGRAPHICS_CROPS_PER_PERSON]

        target_ids = sorted(pooled_crops.keys())

        print(f"Demographics estimation: judging {len(target_ids)} detected person(s), "
              f"up to {app_config.DEMOGRAPHICS_CROPS_PER_PERSON} sampled frame(s) each.")

        for pid in target_ids:
            self.results[pid] = self._estimate_person(pid, pooled_crops[pid])

        return self.results

    def _estimate_person(self, pid, crops):
        votes = {attr: Counter() for attr in self._categories}

        for i, crop in enumerate(crops):
            values = self._ask_vlm(crop, pid, i)
            for attr, value in values.items():
                if value is not None:
                    votes[attr][value] += 1

        threshold = app_config.DEMOGRAPHICS_MAJORITY_THRESHOLD
        result = {'sample_count': len(crops)}
        for attr, counter in votes.items():
            total_votes = sum(counter.values())
            winner, winner_count = counter.most_common(1)[0] if counter else (None, 0)
            # Only report a winner once it clears the majority threshold of
            # this attribute's non-null votes - a bare plurality from a
            # near-tied vote (e.g. 2/2/1 across three categories) is left
            # unresolved rather than confidently reported.
            result[attr] = winner if total_votes > 0 and winner_count > threshold * total_votes else None
            result[f'{attr}_votes'] = dict(counter)
        return result

    def _ask_vlm(self, crop_bytes, pid, crop_index):
        """Returns {attribute: value|None} for one sampled crop, and logs
        the raw exchange to self.query_log."""
        data_url = f"data:image/jpeg;base64,{base64.b64encode(crop_bytes).decode('ascii')}"
        entry = {'person_id': pid, 'crop_index': crop_index}

        try:
            response = self._client.chat.completions.create(
                model=app_config.VLM_MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": app_config.DEMOGRAPHICS_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                timeout=app_config.VLM_REQUEST_TIMEOUT_S,
            )
            text = (response.choices[0].message.content or "").strip()
            values = self._parse_verdict(text)
            entry.update({'response': text, 'error': None, **values})
        except Exception as e:
            values = {attr: None for attr in self._categories}
            entry.update({'response': None, 'error': str(e), **values})

        self.query_log.append(entry)
        return values

    def _parse_verdict(self, text):
        """
        Parses the model's first line, expected as one value per attribute
        (in ATTRIBUTE_CATEGORIES_CONFIG order) separated by commas - e.g.
        'young_adult, female, athletic, black'. Falls back to scanning the
        whole response for a known category word per attribute if the model
        didn't follow the requested format exactly (that fallback is a
        best-effort heuristic - it can't distinguish a mentioned category
        from a negated one, e.g. "not a child, but a teen").
        """
        result = {attr: None for attr in self._categories}
        if not text:
            return result

        first_line = text.lower().splitlines()[0]

        strict = self._parse_strict(first_line)
        if strict is not None:
            return strict

        lowered = text.lower()
        for attr, categories in self._categories.items():
            value = self._find_category(first_line, categories)
            if value is None:
                value = self._find_category(lowered, categories)
            result[attr] = value

        return result

    def _parse_strict(self, first_line):
        """
        Tries to parse `first_line` as exactly one comma-separated token per
        attribute, in ATTRIBUTE_CATEGORIES_CONFIG order, each token matching
        one of that attribute's categories exactly (after collapsing
        internal whitespace to a single underscore, so 'young adult' still
        matches the 'young_adult' category). Returns None - so the caller
        falls back to fuzzy scanning - if the token count doesn't match or
        any token isn't an exact category match, rather than guessing.
        """
        tokens = [re.sub(r"\s+", "_", token.strip()) for token in first_line.split(",")]
        if len(tokens) != len(self._categories):
            return None

        result = {}
        for token, (attr, categories) in zip(tokens, self._categories.items()):
            if token not in categories:
                return None
            result[attr] = token
        return result

    @staticmethod
    def _find_category(text, categories):
        """
        Finds the category from `categories` present in `text` as a whole
        word/phrase - not merely as a substring (so e.g. 'male' won't
        spuriously match inside 'female'). Multi-word categories like
        'young_adult' match either underscore- or space-separated spelling
        ('young_adult' or 'young adult'), since the model isn't guaranteed
        to reuse the exact underscored token from the prompt. Longer
        categories are checked first, so e.g. 'young_adult' always wins over
        'adult' regardless of how the categories happen to be declared.
        """
        for category in sorted(categories, key=len, reverse=True):
            pattern = re.escape(category).replace('_', r'[\s_]+')
            if re.search(rf"(?<![a-z]){pattern}(?![a-z])", text):
                return category
        return None
