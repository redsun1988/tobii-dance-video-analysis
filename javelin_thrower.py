import numpy as np

class JavelinThrower:
    @staticmethod
    def point_in_box(point, box):
        """Checks if a point (x, y) is inside a bounding box (x1, y1, x2, y2)."""
        if box is None:
            return False
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def is_point_in_rotated_rect(p, a, b, width):
        """
        Checks if point p is inside rotated rectangle.

        Arguments:
            p: (x, y) - checked point
            a: (x, y) - first point on central line
            b: (x, y) - second point on central line
            width: rectangle's width in pixels
        """
        # Converting all points to numpy vectors
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)

        # Rectangle center
        center = (a + b) / 2.0

        # Vector from a to b and its length
        ab = b - a
        length = np.linalg.norm(ab)

        if length == 0:
            return False   # invalid case

        # Normalized vector from a to b (OX axis of rectangle)
        ab_dir = ab / length

        # Normalized perpendicular vector (OY axis of rectangle)
        perp_dir = np.array([-ab_dir[1], ab_dir[0]])

        # Translating point p to rectangle's coordinate system (with zero in 'center')
        rel = p - center

        # Projections of the point on rectangle axes
        x_proj = np.dot(rel, ab_dir)
        y_proj = np.dot(rel, perp_dir)

        # Checking central line length constraints (half length) and width constraints (half width)
        half_length = length / 2.0
        half_width = width / 2.0

        return (-half_length <= x_proj <= half_length) and (-half_width <= y_proj <= half_width)

    @staticmethod
    def rotated_rect_corners(a, b, width):
        """
        Returns the 4 corners of the rotated rectangle defined by central
        axis a->b and perpendicular `width`, in order around the perimeter.
        """
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        ab = b - a
        length = np.linalg.norm(ab)
        half_w = width / 2.0

        if length == 0:
            return [
                (a[0] - half_w, a[1] - half_w),
                (a[0] + half_w, a[1] - half_w),
                (a[0] + half_w, a[1] + half_w),
                (a[0] - half_w, a[1] + half_w),
            ]

        ab_dir = ab / length
        perp_dir = np.array([-ab_dir[1], ab_dir[0]])
        offset = perp_dir * half_w

        corners = [a - offset, a + offset, b + offset, b - offset]
        return [(float(c[0]), float(c[1])) for c in corners]

    @staticmethod
    def rotated_rect_axis_aligned_bounds(a, b, width, margin=0.0):
        """
        Returns the axis-aligned (x1, y1, x2, y2) box that encloses the
        rotated rectangle, expanded by `margin` pixels on each side. Used to
        pick a representative crop region around a body-part box (e.g. for
        the VLM attention probe), since image cropping needs an axis-aligned
        region even though the underlying hit-test box is rotated.
        """
        corners = JavelinThrower.rotated_rect_corners(a, b, width)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)