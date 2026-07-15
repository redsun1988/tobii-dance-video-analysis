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