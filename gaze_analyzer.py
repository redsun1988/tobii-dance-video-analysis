import pandas as pd
from openpyxl import load_workbook
from collections import defaultdict
from javelin_thrower import JavelinThrower


class GazeAnalyzer:
    """Analyzes gaze data in relation to detected poses and collects statistics."""
    def __init__(self):
        self.gaze_history = defaultdict(list) # Stores (frame_idx, person_id, body_part, gaze_coords)
        self.gaze_duration_per_person = defaultdict(float) # Total gaze time per person
        self.gaze_duration_per_part = defaultdict(float) # Total gaze time per body part
        self.gaze_transitions = defaultdict(lambda: defaultdict(int)) # Count of transitions from part A to part B
        self._last_gazed_person_part = None # (person_id, body_part_name) of the last gazed item for transition tracking
        self.frame_rate = 30 # Default frame rate, will be updated by application
        self.total_frames = 0 # To calculate total video duration

    def set_frame_rate(self, fps):
        self.frame_rate = fps

    def analyze_frame(self, frame_idx, people_data, gaze_coords):
        """
        Analyzes gaze position against detected people and their body parts.
        Records gaze events and updates statistics.
        """
        self.total_frames = max(self.total_frames, frame_idx + 1)
        gazed_person_id = None
        gazed_body_part_name = None

        for person in people_data:
            # Check if gaze is on the person's overall bounding box
            if JavelinThrower.point_in_box(gaze_coords, person['box']):
                gazed_person_id = person['id']

                # Check specific body parts using their rotated rectangles
                # (each body part is (a, b, width) or None)
                for part_name, rotated_box in person['body_parts'].items():
                    if rotated_box is None:
                        continue
                    a, b, width = rotated_box
                    if JavelinThrower.is_point_in_rotated_rect(gaze_coords, a, b, width):
                        gazed_body_part_name = part_name
                        break # Prioritize specific part over general person box

            if gazed_person_id is not None and gazed_body_part_name is not None:
                break # Gaze found on a person and their part

        # Record gaze event
        self.gaze_history[frame_idx].append({
            'gazed_person_id': gazed_person_id,
            'gazed_body_part': gazed_body_part_name,
            'gaze_coords': gaze_coords
        })

        # Update durations (assuming each frame represents 1/frame_rate seconds)
        time_per_frame = 1.0 / self.frame_rate
        if gazed_person_id is not None:
            self.gaze_duration_per_person[gazed_person_id] += time_per_frame
        if gazed_body_part_name is not None:
            self.gaze_duration_per_part[gazed_body_part_name] += time_per_frame

        # Update transition matrix
        current_gazed_item = (gazed_person_id, gazed_body_part_name)
        if self._last_gazed_person_part and self._last_gazed_person_part != current_gazed_item:
            from_person_id, from_part_name = self._last_gazed_person_part
            to_person_id, to_part_name = current_gazed_item

            # Record transitions between parts if they are on the same person or to a different person
            if from_person_id == to_person_id and from_part_name is not None and to_part_name is not None:
                self.gaze_transitions[from_part_name][to_part_name] += 1
            elif from_person_id != to_person_id:
                # Transition to a different person
                self.gaze_transitions[f"Person {from_person_id}"][f"Person {to_person_id}"] += 1
            elif from_part_name is None and to_part_name is not None:
                # Transition from no gaze to a specific part
                self.gaze_transitions["None"][to_part_name] += 1
            elif from_part_name is not None and to_part_name is None:
                # Transition from a specific part to no gaze
                self.gaze_transitions[from_part_name]["None"] += 1

        self._last_gazed_person_part = current_gazed_item

    def generate_statistics(self):
        """Calculates final statistics after video processing."""
        total_gaze_time = sum(self.gaze_duration_per_person.values())
        if total_gaze_time == 0:
            return {
                'total_gaze_time': 0,
                'part_gaze_ratio': {},
                'person_gaze_duration': {},
                'person_gaze_ratio': {},
                'gaze_transitions_summary': {},
                'most_gazed_person': None
            }

        part_gaze_ratio = {part: duration / total_gaze_time for part, duration in self.gaze_duration_per_part.items()}
        person_gaze_ratio = {pid: duration / total_gaze_time for pid, duration in self.gaze_duration_per_person.items()}

        most_gazed_person_id = max(self.gaze_duration_per_person, key=self.gaze_duration_per_person.get) if self.gaze_duration_per_person else None

        # Summarize transitions
        gaze_transitions_summary = {}
        for from_part, to_parts in self.gaze_transitions.items():
            gaze_transitions_summary[from_part] = {to_part: count for to_part, count in to_parts.items()}

        return {
            'total_gaze_time': total_gaze_time,
            'part_gaze_ratio': part_gaze_ratio,
            'person_gaze_duration': self.gaze_duration_per_person,
            'person_gaze_ratio': person_gaze_ratio,
            'gaze_transitions_summary': gaze_transitions_summary,
            'most_gazed_person': most_gazed_person_id
        }

    def display_statistics(self, stats):
        """Prints the calculated statistics to the console."""
        print("\n--- Gaze Analysis Statistics ---")
        if stats['total_gaze_time'] == 0:
            print("No gaze detected on any person/body part.")
            return

        print(f"Total gaze time on detected objects: {stats['total_gaze_time']:.2f} seconds")

        print("\nGaze Duration per Person (seconds):")
        for pid, duration in stats['person_gaze_duration'].items():
            print(f"  Person {pid}: {duration:.2f}s ({stats['person_gaze_ratio'][pid]:.2%})")

        print("\nMost gazed person:")
        if stats['most_gazed_person'] is not None:
            print(f"  Person {stats['most_gazed_person']} (Note: Gender information not available from sources).")
        else:
            print("  N/A")

        print("\nGaze Ratio per Body Part:")
        for part, ratio in stats['part_gaze_ratio'].items():
            print(f"  {part.replace('_', ' ').title()}: {ratio:.2%}")

        print("\nGaze Transition Dynamics (counts):")
        if not stats['gaze_transitions_summary']:
            print("  No transitions recorded.")
        else:
            for from_part, to_parts in stats['gaze_transitions_summary'].items():
                for to_part, count in to_parts.items():
                    print(f"  From '{from_part.replace('_', ' ').title()}' to '{to_part.replace('_', ' ').title()}': {count} times")
    
    def save_to_excel(self, filename):
        """Saves all possible and interesting statistics to an Excel file."""
        
        stats = self.generate_statistics()
        
        # Create a new DataFrame for each type of statistic
        person_gaze_duration = pd.DataFrame(list(stats['person_gaze_duration'].items()), columns=['Person ID', 'Gaze Duration'])
        part_gaze_ratio = pd.DataFrame(list(stats['part_gaze_ratio'].items()), columns=['Body Part', 'Ratio'])
        
        # For transitions, we need to flatten the dictionary and create a DataFrame from it
        gaze_transitions = []
        for from_part, to_parts in stats['gaze_transitions_summary'].items():
            for to_part, count in to_parts.items():
                gaze_transitions.append({'From': from_part, 'To': to_part, 'Count': count})
        
        gaze_transitions = pd.DataFrame(gaze_transitions)
        
        # Save each DataFrame to a separate sheet in the Excel file
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            person_gaze_duration.to_excel(writer, sheet_name='Gaze Duration per Person', index=False)
            part_gaze_ratio.to_excel(writer, sheet_name='Ratio of Gaze per Body Part', index=False)
            gaze_transitions.to_excel(writer, sheet_name='Gaze Transition Counts', index=False)
            
        print("Statistics saved to Excel file successfully.")