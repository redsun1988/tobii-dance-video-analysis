import os
import re
import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

import pandas as pd

from config import AppConfig
from javelin_thrower import JavelinThrower

app_config = AppConfig()

_PERSON_LABEL_PREFIX_RE = re.compile(r"^Person (\d+)")


def _remap_sum_dict(d, key_fn):
    """Rekeys a {key: numeric_total} dict through key_fn, summing values
    that land on the same new key (e.g. two merged person IDs)."""
    merged = defaultdict(float)
    for k, v in d.items():
        merged[key_fn(k)] += v
    return dict(merged)


class GazeTarget:
    """
    Canonical identity of whatever the gaze currently falls on. Using a
    single label for all bookkeeping (durations, transitions) instead of
    juggling separate person_id/body_part variables removes the old
    "Person None" bug, where a transition involving "gaze is on nobody" was
    handled inconsistently depending on which side of the transition it was
    on.
    """
    KIND_OUTSIDE = "outside"      # gaze left the video window entirely
    KIND_BACKGROUND = "background"  # inside the window, but not on any detected person
    KIND_PERSON = "person"        # on a person's box, but no specific body part matched
    KIND_PART = "part"            # on a specific body part's rotated box

    __slots__ = ("kind", "person_id", "part_name")

    def __init__(self, kind, person_id=None, part_name=None):
        self.kind = kind
        self.person_id = person_id
        self.part_name = part_name

    @classmethod
    def outside(cls):
        return cls(cls.KIND_OUTSIDE)

    @classmethod
    def background(cls):
        return cls(cls.KIND_BACKGROUND)

    @classmethod
    def person(cls, person_id):
        return cls(cls.KIND_PERSON, person_id=person_id)

    @classmethod
    def part(cls, person_id, part_name):
        return cls(cls.KIND_PART, person_id=person_id, part_name=part_name)

    @property
    def label(self):
        if self.kind == self.KIND_OUTSIDE:
            return "Outside Window"
        if self.kind == self.KIND_BACKGROUND:
            return "Background"
        if self.kind == self.KIND_PERSON:
            return f"Person {self.person_id}"
        return f"Person {self.person_id} - {self.part_name}"

    def __eq__(self, other):
        return isinstance(other, GazeTarget) and self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return f"GazeTarget({self.label!r})"


@dataclass
class FrameAnalysis:
    """Result of analyzing a single frame's gaze sample."""
    target: GazeTarget
    is_saccade: bool
    is_new_fixation: bool
    # Present only when is_new_fixation is True - gives the caller enough
    # geometry to crop a representative image region around the new target
    # for the VLM attention probe.
    crop_hint: Optional[dict]


class GazeAnalyzer:
    """Analyzes gaze data in relation to detected poses and collects statistics."""

    # PersonDemographicsEstimator attribute -> short display label. Kept short
    # (rather than e.g. "Dominant Color") so combinations like "Body Part Gaze
    # by <label>" stay within Excel's 31-character sheet name limit.
    DEMOGRAPHIC_GROUP_LABELS = {
        'gender': 'Gender',
        'age': 'Age',
        'body_build': 'Build',
        'dominant_color': 'Color',
    }

    def __init__(self):
        self.gaze_events = []  # one dict per analyzed frame, for CSV export

        self.gaze_duration_per_label = defaultdict(float)
        self.gaze_duration_per_person = defaultdict(float)
        self.gaze_duration_per_part = defaultdict(float)
        self.gaze_duration_per_person_part = defaultdict(float)  # (person_id, part_name) -> duration, for demographic breakdowns
        self.gaze_duration_per_source = defaultdict(float)
        self.gaze_transitions = defaultdict(lambda: defaultdict(int))

        self.saccade_count = 0
        self.confirmed_fixation_count = 0
        self.total_frames = 0
        self.fixated_person_ids = set()  # person ids that received at least one confirmed fixation

        self._last_target = None            # target recorded on the previous frame (for transitions)
        self._prev_local_gaze = None         # local gaze point on the previous in-window frame (for saccade detection)
        self._prev_time = None               # timestamp of that previous in-window sample

        self._pending_target = None          # candidate new fixation target, awaiting debounce confirmation
        self._pending_count = 0
        self._confirmed_target = None        # last target that was actually confirmed as a new fixation

    def analyze_frame(self, frame_idx, timestamp, people_data, gaze_absolute, gaze_local, in_window, gaze_source, dt):
        """
        Analyzes one gaze sample against the detected people/body parts.

        Arguments:
            frame_idx: index of the processed frame
            timestamp: seconds since analysis start (monotonic, real wall-clock)
            people_data: output of PoseEstimator.process_frame
            gaze_absolute: (x, y) gaze in absolute desktop coordinates
            gaze_local: (x, y) gaze in the video window's client coordinates, or None if in_window is False
            in_window: whether the gaze currently falls inside the video window's client area
            gaze_source: "tobii" or "simulated"
            dt: real elapsed seconds since the previous analyzed frame (used for duration accounting,
                since the actual processing rate is not a fixed FPS)

        Returns a FrameAnalysis describing the resolved target, whether this
        sample was part of a saccade, and whether it confirms a new fixation
        (in which case crop_hint carries the geometry needed to grab a
        representative crop for the VLM probe).
        """
        self.total_frames = max(self.total_frames, frame_idx + 1)

        if not in_window or gaze_local is None:
            target = GazeTarget.outside()
            crop_hint = None
        else:
            target, crop_hint = self._resolve_target(people_data, gaze_local)

        is_saccade = self._update_saccade_state(gaze_local, timestamp, in_window)
        is_new_fixation = self._update_fixation_state(target, is_saccade)

        self._accumulate_durations(target, gaze_source, dt)
        self._record_transition(target)

        self.gaze_events.append({
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'gaze_x_abs': gaze_absolute[0] if gaze_absolute else None,
            'gaze_y_abs': gaze_absolute[1] if gaze_absolute else None,
            'gaze_x_local': gaze_local[0] if gaze_local else None,
            'gaze_y_local': gaze_local[1] if gaze_local else None,
            'in_window': in_window,
            'gaze_source': gaze_source,
            'target_label': target.label,
            'is_saccade': is_saccade,
            'is_new_fixation': is_new_fixation,
        })

        return FrameAnalysis(
            target=target,
            is_saccade=is_saccade,
            is_new_fixation=is_new_fixation,
            crop_hint=crop_hint if is_new_fixation else None,
        )

    def _resolve_target(self, people_data, gaze_local):
        """Finds what the gaze currently falls on: a body part, a person's
        general box, or the background. Returns (GazeTarget, crop_hint)."""
        for person in people_data:
            if not JavelinThrower.point_in_box(gaze_local, person['box']):
                continue

            for part_name, rotated_box in person['body_parts'].items():
                if rotated_box is None:
                    continue
                a, b, width = rotated_box
                if JavelinThrower.is_point_in_rotated_rect(gaze_local, a, b, width):
                    crop_hint = {
                        'kind': 'part',
                        'rotated_box': rotated_box,
                        'person_box': person['box'],
                        'gaze_point': gaze_local,
                    }
                    return GazeTarget.part(person['id'], part_name), crop_hint

            crop_hint = {
                'kind': 'person',
                'rotated_box': None,
                'person_box': person['box'],
                'gaze_point': gaze_local,
            }
            return GazeTarget.person(person['id']), crop_hint

        crop_hint = {'kind': 'background', 'rotated_box': None, 'person_box': None, 'gaze_point': gaze_local}
        return GazeTarget.background(), crop_hint

    def _update_saccade_state(self, gaze_local, timestamp, in_window):
        """Flags a saccade when the gaze jumps far/fast enough since the
        last in-window sample. Gaze samples that leave the window don't
        reset the reference point (a jump back in doesn't look like a
        teleport), they're simply not compared."""
        is_saccade = False

        if in_window and gaze_local is not None:
            if self._prev_local_gaze is not None and self._prev_time is not None:
                dx = gaze_local[0] - self._prev_local_gaze[0]
                dy = gaze_local[1] - self._prev_local_gaze[1]
                distance = math.hypot(dx, dy)
                elapsed = max(timestamp - self._prev_time, 1e-6)
                velocity = distance / elapsed

                if distance >= app_config.SACCADE_MIN_DISTANCE_PX or velocity >= app_config.SACCADE_MIN_VELOCITY_PX_S:
                    is_saccade = True
                    self.saccade_count += 1

            self._prev_local_gaze = gaze_local
            self._prev_time = timestamp

        return is_saccade

    def _update_fixation_state(self, target, is_saccade):
        """Debounces saccade landings: only treat a post-saccade target as a
        confirmed new fixation once the gaze has stayed on it for
        FIXATION_CONFIRM_SAMPLES consecutive samples, and only once per
        distinct target (so it doesn't keep re-firing every frame)."""
        if is_saccade:
            self._pending_target = target
            self._pending_count = 1
        elif self._pending_target is not None and self._pending_target == target:
            self._pending_count += 1
        elif self._pending_target is not None:
            self._pending_target = None
            self._pending_count = 0

        is_new_fixation = False
        if (self._pending_target is not None
                and self._pending_count >= app_config.FIXATION_CONFIRM_SAMPLES
                and self._pending_target != self._confirmed_target):
            is_new_fixation = True
            self.confirmed_fixation_count += 1
            self._confirmed_target = self._pending_target
            if self._confirmed_target.kind in (GazeTarget.KIND_PERSON, GazeTarget.KIND_PART):
                self.fixated_person_ids.add(self._confirmed_target.person_id)
            self._pending_target = None
            self._pending_count = 0

        return is_new_fixation

    def _accumulate_durations(self, target, gaze_source, dt):
        self.gaze_duration_per_label[target.label] += dt
        self.gaze_duration_per_source[gaze_source] += dt

        if target.kind in (GazeTarget.KIND_PERSON, GazeTarget.KIND_PART):
            self.gaze_duration_per_person[target.person_id] += dt
        if target.kind == GazeTarget.KIND_PART:
            self.gaze_duration_per_part[target.part_name] += dt
            self.gaze_duration_per_person_part[(target.person_id, target.part_name)] += dt

    def _record_transition(self, target):
        if self._last_target is not None and self._last_target != target:
            self.gaze_transitions[self._last_target.label][target.label] += 1
        self._last_target = target

    def remap_person_ids(self, id_map):
        """
        Applies a {track_id: canonical_id} mapping (from post-hoc identity
        reconciliation) to every place a person id was recorded, merging
        durations/counts for ids that turned out to be the same real
        person. Must run after the video ends, before generate_statistics()
        or any export.
        """
        id_map = {int(old): int(new) for old, new in id_map.items() if int(old) != int(new)}
        if not id_map:
            return

        def relabel(label):
            m = _PERSON_LABEL_PREFIX_RE.match(label)
            if not m or int(m.group(1)) not in id_map:
                return label
            return _PERSON_LABEL_PREFIX_RE.sub(f"Person {id_map[int(m.group(1))]}", label, count=1)

        self.gaze_duration_per_person = _remap_sum_dict(
            self.gaze_duration_per_person, lambda pid: id_map.get(pid, pid)
        )
        self.gaze_duration_per_person_part = _remap_sum_dict(
            self.gaze_duration_per_person_part, lambda key: (id_map.get(key[0], key[0]), key[1])
        )
        self.gaze_duration_per_label = _remap_sum_dict(self.gaze_duration_per_label, relabel)

        new_transitions = defaultdict(lambda: defaultdict(int))
        for from_label, to_labels in self.gaze_transitions.items():
            new_from = relabel(from_label)
            for to_label, count in to_labels.items():
                new_transitions[new_from][relabel(to_label)] += count
        self.gaze_transitions = new_transitions

        for event in self.gaze_events:
            event['target_label'] = relabel(event['target_label'])

        print(f"Applied identity reconciliation: {len(id_map)} person ID(s) merged -> {id_map}")

    def generate_statistics(self, demographics=None):
        """
        Calculates final statistics after video processing. When
        `demographics` (PersonDemographicsEstimator.estimate() output,
        keyed by canonical person ID) is given, also aggregates gaze
        duration/ratio and body-part gaze ratio by the gazed-at person's
        gender and age category, instead of by individual person ID.
        """
        total_gaze_time = sum(self.gaze_duration_per_person.values())  # time spent on any person (box or part)
        total_time = sum(self.gaze_duration_per_label.values())        # time across the whole session, incl. background/outside

        part_gaze_ratio = {
            part: duration / total_gaze_time for part, duration in self.gaze_duration_per_part.items()
        } if total_gaze_time > 0 else {}
        person_gaze_ratio = {
            pid: duration / total_gaze_time for pid, duration in self.gaze_duration_per_person.items()
        } if total_gaze_time > 0 else {}
        label_gaze_ratio = {
            label: duration / total_time for label, duration in self.gaze_duration_per_label.items()
        } if total_time > 0 else {}

        most_gazed_person = max(self.gaze_duration_per_person, key=self.gaze_duration_per_person.get) \
            if self.gaze_duration_per_person else None
        most_gazed_part = max(self.gaze_duration_per_part, key=self.gaze_duration_per_part.get) \
            if self.gaze_duration_per_part else None

        gaze_transitions_summary = {
            from_label: dict(to_labels) for from_label, to_labels in self.gaze_transitions.items()
        }

        stats = {
            'total_gaze_time': total_gaze_time,
            'total_time': total_time,
            'part_gaze_ratio': part_gaze_ratio,
            'part_gaze_duration': dict(self.gaze_duration_per_part),
            'person_gaze_duration': dict(self.gaze_duration_per_person),
            'person_gaze_ratio': person_gaze_ratio,
            'label_gaze_duration': dict(self.gaze_duration_per_label),
            'label_gaze_ratio': label_gaze_ratio,
            'gaze_duration_per_source': dict(self.gaze_duration_per_source),
            'gaze_transitions_summary': gaze_transitions_summary,
            'most_gazed_person': most_gazed_person,
            'most_gazed_part': most_gazed_part,
            'saccade_count': self.saccade_count,
            'confirmed_fixation_count': self.confirmed_fixation_count,
            'total_frames': self.total_frames,
        }

        if demographics:
            stats.update(self._demographic_gaze_stats(demographics, total_gaze_time))

        return stats

    def _demographic_gaze_stats(self, demographics, total_gaze_time):
        """
        Aggregates gaze duration and body-part gaze ratio by each of the
        gazed-at person's estimated appearance attributes - gender, age,
        body build, dominant clothing color (from
        PersonDemographicsEstimator.estimate()) - instead of by individual
        person ID. A person with no resolved value for an attribute
        (demographics disabled for them, or the VLM votes never reached a
        majority) falls into an 'unknown' bucket rather than being silently
        dropped.
        """
        def group_of(pid, attr):
            info = demographics.get(pid)
            return (info.get(attr) if info else None) or 'unknown'

        def ratio_of_total(d):
            return {k: (v / total_gaze_time if total_gaze_time > 0 else 0.0) for k, v in d.items()}

        def ratio_within_group(nested, group_totals):
            # Divides by the group's true total gaze duration (which also
            # includes generic person-box gaze that never matched a specific
            # part) - not by sum(parts.values()), which would only cover the
            # part-attributed subtotal and overstate every part's ratio.
            result = {}
            for group, parts in nested.items():
                group_total = group_totals.get(group, 0.0)
                result[group] = {p: (v / group_total if group_total > 0 else 0.0) for p, v in parts.items()}
            return result

        stats = {}
        for attr in self.DEMOGRAPHIC_GROUP_LABELS:
            duration = defaultdict(float)
            for pid, d in self.gaze_duration_per_person.items():
                duration[group_of(pid, attr)] += d

            part_duration = defaultdict(lambda: defaultdict(float))
            for (pid, part), d in self.gaze_duration_per_person_part.items():
                part_duration[group_of(pid, attr)][part] += d

            duration = dict(duration)
            stats[f'{attr}_gaze_duration'] = duration
            stats[f'{attr}_gaze_ratio'] = ratio_of_total(duration)
            stats[f'{attr}_part_gaze_duration'] = {g: dict(p) for g, p in part_duration.items()}
            stats[f'{attr}_part_gaze_ratio'] = ratio_within_group(part_duration, duration)
            stats[f'most_gazed_{attr}'] = max(duration, key=duration.get) if duration else None

        return stats

    def display_statistics(self, stats, demographics=None):
        """Prints the calculated statistics to the console."""
        print("\n--- Gaze Analysis Statistics ---")
        print(f"Total frames analyzed: {stats['total_frames']}")
        print(f"Total session time: {stats['total_time']:.2f} seconds")
        print(f"Saccades detected: {stats['saccade_count']}, confirmed new fixations: {stats['confirmed_fixation_count']}")

        if stats['gaze_duration_per_source']:
            print("\nGaze source breakdown (seconds):")
            for source, duration in stats['gaze_duration_per_source'].items():
                print(f"  {source}: {duration:.2f}s")

        if stats['total_gaze_time'] == 0:
            print("\nNo gaze detected on any person/body part.")
            return

        print(f"\nTotal gaze time on detected people: {stats['total_gaze_time']:.2f} seconds")

        print("\nGaze Duration per Person (seconds):")
        for pid, duration in stats['person_gaze_duration'].items():
            demo = (demographics or {}).get(pid)
            demo_suffix = (
                f" - {demo.get('age') or 'age?'}, {demo.get('gender') or 'gender?'}, "
                f"{demo.get('body_build') or 'build?'}, {demo.get('dominant_color') or 'color?'}"
            ) if demo else ""
            print(f"  Person {pid}: {duration:.2f}s ({stats['person_gaze_ratio'][pid]:.2%}){demo_suffix}")

        print("\nMost gazed person:")
        print(f"  Person {stats['most_gazed_person']}" if stats['most_gazed_person'] is not None else "  N/A")

        print("\nGaze Ratio per Body Part (of time spent on any person):")
        for part, ratio in sorted(stats['part_gaze_ratio'].items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {part.replace('_', ' ').title()}: {ratio:.2%}")

        print("\nMost gazed body part:")
        print(f"  {stats['most_gazed_part'].replace('_', ' ').title()}" if stats['most_gazed_part'] else "  N/A")

        if 'gender_gaze_duration' in stats:
            for attr, label in self.DEMOGRAPHIC_GROUP_LABELS.items():
                print(f"\nGaze Duration by {label} (seconds):")
                for group, duration in stats[f'{attr}_gaze_duration'].items():
                    ratio = stats[f'{attr}_gaze_ratio'][group]
                    print(f"  {group}: {duration:.2f}s ({ratio:.2%})")

                print(f"\nBody Part Gaze Ratio by {label} (of time spent on that group):")
                for group, parts in stats[f'{attr}_part_gaze_ratio'].items():
                    print(f"  {group}:")
                    for part, ratio in sorted(parts.items(), key=lambda kv: kv[1], reverse=True):
                        print(f"    {part.replace('_', ' ').title()}: {ratio:.2%}")

        print("\nGaze Transition Dynamics (counts):")
        if not stats['gaze_transitions_summary']:
            print("  No transitions recorded.")
        else:
            for from_label, to_labels in stats['gaze_transitions_summary'].items():
                for to_label, count in to_labels.items():
                    print(f"  From '{from_label}' to '{to_label}': {count} times")

    def save_to_excel(self, filename, demographics=None, stats=None):
        """Saves all possible and interesting statistics to an Excel file.
        Pass a precomputed `stats` (from generate_statistics()) to avoid
        recomputing it when the caller already has one; otherwise it's
        computed here from `demographics`."""
        if stats is None:
            stats = self.generate_statistics(demographics)

        person_gaze_duration = pd.DataFrame(list(stats['person_gaze_duration'].items()), columns=['Person ID', 'Gaze Duration'])
        part_gaze_ratio = pd.DataFrame(list(stats['part_gaze_ratio'].items()), columns=['Body Part', 'Ratio'])

        gaze_transitions = []
        for from_label, to_labels in stats['gaze_transitions_summary'].items():
            for to_label, count in to_labels.items():
                gaze_transitions.append({'From': from_label, 'To': to_label, 'Count': count})
        gaze_transitions = pd.DataFrame(gaze_transitions)

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            person_gaze_duration.to_excel(writer, sheet_name='Gaze Duration per Person', index=False)
            part_gaze_ratio.to_excel(writer, sheet_name='Ratio of Gaze per Body Part', index=False)
            gaze_transitions.to_excel(writer, sheet_name='Gaze Transition Counts', index=False)

            if demographics:
                self._demographics_dataframe(demographics).to_excel(writer, sheet_name='Person Demographics', index=False)

            if 'gender_gaze_duration' in stats:
                for attr, label in self.DEMOGRAPHIC_GROUP_LABELS.items():
                    self._group_duration_dataframe(stats[f'{attr}_gaze_duration'], stats[f'{attr}_gaze_ratio'], attr) \
                        .to_excel(writer, sheet_name=f'Gaze Duration by {label}', index=False)
                    self._group_part_dataframe(stats[f'{attr}_part_gaze_duration'], stats[f'{attr}_part_gaze_ratio'], attr) \
                        .to_excel(writer, sheet_name=f'Body Part Gaze by {label}', index=False)

        print(f"Statistics saved to Excel file: {filename}")

    @staticmethod
    def _demographics_dataframe(demographics):
        """Builds the per-person appearance majority-vote table (age,
        gender, body build, dominant clothing color) shared by the Excel and
        CSV exports."""
        rows = [
            {
                'person_id': pid,
                'age': result.get('age'),
                'gender': result.get('gender'),
                'body_build': result.get('body_build'),
                'dominant_color': result.get('dominant_color'),
                'sample_count': result.get('sample_count'),
                'age_votes': result.get('age_votes'),
                'gender_votes': result.get('gender_votes'),
                'body_build_votes': result.get('body_build_votes'),
                'dominant_color_votes': result.get('dominant_color_votes'),
            }
            for pid, result in (demographics or {}).items()
        ]
        return pd.DataFrame(rows, columns=[
            'person_id', 'age', 'gender', 'body_build', 'dominant_color', 'sample_count',
            'age_votes', 'gender_votes', 'body_build_votes', 'dominant_color_votes',
        ])

    @staticmethod
    def _group_duration_dataframe(duration_by_group, ratio_by_group, group_col):
        """Builds a {gender|age -> gaze duration/ratio} table, shared by the
        Excel and CSV exports."""
        rows = [
            {group_col: group, 'duration_s': duration, 'ratio': ratio_by_group.get(group, 0.0)}
            for group, duration in (duration_by_group or {}).items()
        ]
        return pd.DataFrame(rows, columns=[group_col, 'duration_s', 'ratio'])

    @staticmethod
    def _group_part_dataframe(duration_by_group_part, ratio_by_group_part, group_col):
        """Builds a {gender|age -> {body_part -> gaze duration/ratio}} table
        (flattened to one row per group/part pair), shared by the Excel and
        CSV exports."""
        rows = [
            {
                group_col: group,
                'body_part': part,
                'duration_s': duration,
                'ratio': ratio_by_group_part.get(group, {}).get(part, 0.0),
            }
            for group, parts in (duration_by_group_part or {}).items()
            for part, duration in parts.items()
        ]
        return pd.DataFrame(rows, columns=[group_col, 'body_part', 'duration_s', 'ratio'])

    def save_to_csv(self, output_dir, vlm_query_log=None, identity_comparison_log=None,
                     demographics=None, demographics_query_log=None, stats=None):
        """
        Saves all collected statistics to a set of CSV files for further
        analytics: per-frame gaze events, durations, transitions, the VLM
        attention-probe query log, the identity-reconciliation comparison
        log, the per-person age/gender majority-vote results and their
        underlying per-crop judgments, and a one-row summary.

        Pass a precomputed `stats` (from generate_statistics()) to avoid
        recomputing it when the caller already has one; otherwise it's
        computed here from `demographics`.
        """
        os.makedirs(output_dir, exist_ok=True)
        if stats is None:
            stats = self.generate_statistics(demographics)

        pd.DataFrame(self.gaze_events).to_csv(os.path.join(output_dir, 'gaze_events.csv'), index=False)

        pd.DataFrame(
            list(stats['person_gaze_duration'].items()), columns=['person_id', 'duration_s']
        ).to_csv(os.path.join(output_dir, 'person_duration.csv'), index=False)

        pd.DataFrame(
            list(stats['part_gaze_duration'].items()), columns=['body_part', 'duration_s']
        ).to_csv(os.path.join(output_dir, 'body_part_duration.csv'), index=False)

        transitions_rows = [
            {'from_label': from_label, 'to_label': to_label, 'count': count}
            for from_label, to_labels in stats['gaze_transitions_summary'].items()
            for to_label, count in to_labels.items()
        ]
        pd.DataFrame(transitions_rows, columns=['from_label', 'to_label', 'count']).to_csv(
            os.path.join(output_dir, 'gaze_transitions.csv'), index=False
        )

        if vlm_query_log is not None:
            pd.DataFrame(vlm_query_log).to_csv(os.path.join(output_dir, 'vlm_queries.csv'), index=False)

        if identity_comparison_log is not None:
            pd.DataFrame(identity_comparison_log).to_csv(
                os.path.join(output_dir, 'identity_reconciliation.csv'), index=False
            )

        if demographics:
            self._demographics_dataframe(demographics).to_csv(
                os.path.join(output_dir, 'person_demographics.csv'), index=False
            )

        if demographics_query_log:
            pd.DataFrame(demographics_query_log).to_csv(
                os.path.join(output_dir, 'demographics_queries.csv'), index=False
            )

        if 'gender_gaze_duration' in stats:
            for attr in self.DEMOGRAPHIC_GROUP_LABELS:
                self._group_duration_dataframe(stats[f'{attr}_gaze_duration'], stats[f'{attr}_gaze_ratio'], attr).to_csv(
                    os.path.join(output_dir, f'gaze_by_{attr}.csv'), index=False
                )
                self._group_part_dataframe(stats[f'{attr}_part_gaze_duration'], stats[f'{attr}_part_gaze_ratio'], attr).to_csv(
                    os.path.join(output_dir, f'body_part_by_{attr}.csv'), index=False
                )

        summary_row = {
            'total_frames': stats['total_frames'],
            'total_time_s': stats['total_time'],
            'total_gaze_on_people_s': stats['total_gaze_time'],
            'saccade_count': stats['saccade_count'],
            'confirmed_fixation_count': stats['confirmed_fixation_count'],
            'most_gazed_person': stats['most_gazed_person'],
            'most_gazed_part': stats['most_gazed_part'],
            'vlm_query_count': len(vlm_query_log) if vlm_query_log is not None else 0,
            'identity_comparison_count': len(identity_comparison_log) if identity_comparison_log is not None else 0,
            'demographics_person_count': len(demographics) if demographics is not None else 0,
            'demographics_query_count': len(demographics_query_log) if demographics_query_log is not None else 0,
        }
        for attr in self.DEMOGRAPHIC_GROUP_LABELS:
            summary_row[f'most_gazed_{attr}'] = stats.get(f'most_gazed_{attr}')
        for source, duration in stats['gaze_duration_per_source'].items():
            summary_row[f'time_source_{source}_s'] = duration

        pd.DataFrame([summary_row]).to_csv(os.path.join(output_dir, 'summary.csv'), index=False)

        print(f"Statistics saved to CSV files in: {output_dir}")
