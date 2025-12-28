"""Shape class for representing and manipulating polygon ROIs with JAX-accelerated operations."""

from __future__ import annotations
from typing import List, Tuple, Optional, Callable, Any

import jax.numpy as jnp
from jax import vmap
import cv2 as cv
import numpy as np
from .Point import Point

confirm_box = False
draw_rectangle = False
box_x = 0
box_y = 0
box_w = 0
box_h = 0
k = 0
p = None


class Shape:
    """A polygon shape defined by a list of points.

    This class provides methods for shape manipulation, transformation,
    and ROI (Region of Interest) extraction. All mathematical operations
    use JAX numpy for potential GPU acceleration.

    Attributes:
        points: List of Point objects defining the shape vertices.
        center: The centroid of the shape as a Point object.
        min_x: Minimum x-coordinate (set by to_rectangle).
        max_x: Maximum x-coordinate (set by to_rectangle).
        min_y: Minimum y-coordinate (set by to_rectangle).
        max_y: Maximum y-coordinate (set by to_rectangle).
        width: Width of bounding box (set by to_rectangle).
        height: Height of bounding box (set by to_rectangle).
        histogram: RGB histogram (set by get_histogram).
    """

    def __init__(self, points: List[Tuple[float, float]]) -> None:
        """Initialize a Shape with a list of points.

        Args:
            points: List of (x, y) tuples defining the shape vertices.
        """
        self.points: List[Point] = []
        for p in points:
            point = Point(*p)
            self.points.append(point)
        self.centroid()

    def to_image(self, i: int, frame: np.ndarray) -> None:
        """Save the extracted content of the shape to an image file.

        Args:
            i: Index to use in the filename.
            frame: The image frame to extract content from.
        """
        cv.imwrite('image{}.jpg'.format(i), self.extract_content(frame))

    def to_array(self) -> jnp.ndarray:
        """Convert shape points to a JAX array.

        Returns:
            A JAX array of shape (n_points, 2) containing the coordinates.
        """
        return jnp.array([jnp.array(p.to_tuple()) for p in self.points])

    def centroid(self) -> None:
        """Calculate and update the centroid of the shape."""
        points = jnp.array([jnp.array(p.to_tuple())
                        for p in self.points]) / len(self.points)
        self.center = Point(*points.sum(axis=0))

    def translate_x(self, x: float) -> None:
        """Translate all points along the x-axis.

        Args:
            x: The amount to translate in the x direction.
        """
        for point in self.points:
            point.translate_x(x)

    def translate_y(self, y: float) -> None:
        """Translate all points along the y-axis.

        Args:
            y: The amount to translate in the y direction.
        """
        for point in self.points:
            point.translate_y(y)

    def translate_to(self, x: float, y: float) -> None:
        """Translate the shape so the bounding box max corner is at (x, y).

        Args:
            x: The target x-coordinate for the max corner.
            y: The target y-coordinate for the max corner.
        """
        point = self.points[0].to_tuple()
        self.to_rectangle()

        xx = self.max_x
        yy = self.max_y
        x_distance = x - xx
        y_distance = y - yy
        for p in self.points:
            p.translate_x(x_distance)
            p.translate_y(y_distance)
        self.centroid()

    def rotate_around_center(self, theta: float) -> None:
        """Rotate the shape around its centroid.

        Args:
            theta: The angle to rotate in radians.
        """
        P = jnp.array([jnp.array(list(p.to_tuple())) for p in self.points])
        self.centroid()
        C = jnp.array(self.center.to_tuple())
        C = jnp.tile(C, (len(self.points), 1))
        R = jnp.array([[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]])
        P_res = jnp.dot(R, (P - C).T) + C.T
        for i, p in zip(range(len(self.points)), self.points):
            p.x = float(P_res[0, i])
            p.y = float(P_res[1, i])
            p.round_point()
        self.centroid()

    def reposition(self, x: float, y: float) -> None:
        """Reposition the shape (not implemented).

        Args:
            x: The target x-coordinate.
            y: The target y-coordinate.
        """
        pass

    def update(self, x: float, y: float, theta: float) -> None:
        """Apply rotation and translation to the shape.

        Args:
            x: The amount to translate in the x direction.
            y: The amount to translate in the y direction.
            theta: The angle to rotate in radians.
        """
        self.rotate_around_center(theta)
        self.translate_x(x)
        self.translate_y(y)

    def to_rectangle(self) -> List[Tuple[float, float]]:
        """Calculate the bounding box of the shape.

        Creates a bounding box from the current Shape's vertices and stores
        min/max coordinates and dimensions as instance attributes.

        Returns:
            A list containing two tuples: [(min_x, min_y), (max_x, max_y)].

        Note:
            Cite: https://stackoverflow.com/a/30902423/6512445
        """
        # extract all the points
        pts = self.to_array()
        # find the maximum x
        xs = pts[:, 0]
        ys = pts[:, 1]
        self.max_x = jnp.amax(xs)
        # find the minimum x
        self.min_x = jnp.amin(xs)
        # find the maximum y
        self.max_y = jnp.amax(ys)
        # find the minimum y
        self.min_y = jnp.amin(ys)
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y
        # return frame[min_x:max_x, max_y:min_y]
        return [(float(self.min_x), float(self.min_y)), (float(self.max_x), float(self.max_y))]

    def extract_content(self, frame: np.ndarray) -> np.ndarray:
        """Extract the ROI content from an image frame.

        Creates a mask from the shape polygon and applies it to extract
        only the pixels inside the shape.

        Args:
            frame: The image frame (numpy array) to extract from.

        Returns:
            A numpy array containing the masked image with only ROI pixels.
        """
        mask = np.zeros(frame.shape, dtype=np.uint8)
        roi_corners = np.array(
            [[p.to_tuple() for p in self.points]], dtype=np.int32)
        # fill the ROI so it doesn't get wiped out when the mask is applied
        channel_count = frame.shape[2]

        # get rid of 0 as to difer them from the mask 0
        # which may cause the mask to wipe them out
        # when performing the bitwise and
        frame[np.where(frame == 0)] = 1
        # fill the wanted pixels with white in mask
        ignore_mask_color = (255,)*channel_count
        cv.fillPoly(mask, roi_corners, ignore_mask_color)

        # keep the pixels with True
        masked_image = cv.bitwise_and(frame, mask)
        return masked_image

    def draw_shape(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 1
    ) -> None:
        """Draw the shape outline on an image frame.

        Args:
            frame: The image frame (numpy array) to draw on.
            color: BGR color tuple for the lines (default: yellow).
            thickness: Line thickness in pixels (default: 1).
        """
        for i in range(len(self.points)-1):
            self.draw_line(self.points[i].to_tuple(
            ), self.points[i+1].to_tuple(), frame, color, thickness)
        # drawing the last line between the first point
        # and the last point of the shape
        self.draw_line(self.points[0].to_tuple(), self.points[len(
            self.points) - 1].to_tuple(), frame, color, thickness)

    def get_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate the RGB histogram of the shape's content.

        Args:
            frame: The image frame to calculate histogram from.

        Returns:
            A numpy array of shape (3, 255) containing R, G, B histograms.
        """
        content = self.extract_content(frame)

        histogram_b = cv.calcHist([content], [
                                  0], None, [255], [0, 256]).flatten()
        histogram_g = cv.calcHist([content], [
                                  1], None, [255], [0, 256]).flatten()
        histogram_r = cv.calcHist([content], [
                                  2], None, [255], [0, 256]).flatten()
        histogram = np.array([histogram_r, histogram_g, histogram_b], dtype=int)
        self.histogram = histogram
        return histogram

    def draw_line(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        frame: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int
    ) -> None:
        """Draw a line between two points on a frame.

        Args:
            p1: Start point as (x, y) tuple.
            p2: End point as (x, y) tuple.
            frame: The image frame to draw on.
            color: BGR color tuple.
            thickness: Line thickness in pixels.
        """
        # Convert to integers for OpenCV
        p1_int = (int(p1[0]), int(p1[1]))
        p2_int = (int(p2[0]), int(p2[1]))
        cv.line(frame, p1_int, p2_int, color, thickness)

    @classmethod
    def get_roi(cls, frame: np.ndarray) -> Optional[Shape]:
        """Interactively select a ROI from an image frame.

        This method pops up a window where the user can click to define
        polygon vertices. Press 'c' to confirm the selection.

        Args:
            frame: The image frame (numpy array) to select from.

        Returns:
            A Shape object containing the selected polygon, or None if cancelled.
        """
        global confirm_box, draw_rectangle, k, p
        # global k
        s = None

        cv.namedWindow("Tracker")

        tmp = frame.copy()
        cv.setMouseCallback("Tracker", cls.selectTarget, tmp)
        while not confirm_box:
            tmp2 = tmp.copy()
            if draw_rectangle:
                p.draw_point(tmp2)
            if k == 1 and s is None:
                s = Shape([p.to_tuple()])

            # prevent the addition of multiple instances of the same point
            if k > 1 and len(s.points)+1 == k:
                s.points.append(p)

            if k > 1:
                s.draw_shape(tmp2)

            cv.imshow("Tracker", tmp2)
            if cv.waitKey(30) == ord('c'):
                return s

    @classmethod
    def selectTarget(
        cls,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: np.ndarray
    ) -> None:
        """OpenCV callback function for mouse events during ROI selection.

        Args:
            event: The OpenCV mouse event type.
            x: The x-coordinate of the mouse.
            y: The y-coordinate of the mouse.
            flags: Additional flags from OpenCV.
            param: The image frame to draw on.
        """
        global box_x, box_y, box_w, box_h, confirm_box, draw_rectangle, k, p
        # number of points created
        if event == cv.EVENT_LBUTTONUP:
            p = Point(x, y)
            p.draw_point(param)
            k += 1
        return

    @classmethod
    def copy(cls, shape: Shape) -> Shape:
        """Create a deep copy of a Shape.

        Args:
            shape: The Shape to copy.

        Returns:
            A new Shape object with the same points.
        """
        points = [p.to_tuple() for p in shape.points]
        s = cls(points)
        return s

    def __str__(self) -> str:
        """Return a string representation of the shape.

        Returns:
            A string listing all points in the shape.
        """
        s = 'Printing the shapes points: \n'
        for p in self.points:
            s += str(p)
        return s

    # JAX-enabled parallel processing methods
    @staticmethod
    @jnp.vectorize
    def _vectorized_rotation(
        points_array: jnp.ndarray,
        theta: float,
        center: jnp.ndarray
    ) -> jnp.ndarray:
        """Vectorized rotation operation for JAX parallel processing.

        Args:
            points_array: Array of points to rotate.
            theta: Rotation angle in radians.
            center: Center point for rotation.

        Returns:
            Rotated points array.
        """
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        R = jnp.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
        centered_points = points_array - center
        return jnp.dot(R, centered_points.T).T + center

    @classmethod
    def process_multiple_shapes_parallel(
        cls,
        shapes: List[Shape],
        operation: str,
        *args: Any
    ) -> List[Shape]:
        """Process multiple shapes in parallel using JAX vmap.

        Args:
            shapes: List of Shape objects to process.
            operation: Operation to perform ('rotate', 'translate').
            *args: Arguments for the operation:
                - For 'rotate': theta (rotation angle in radians)
                - For 'translate': dx, dy (translation amounts)

        Returns:
            List of processed Shape objects.

        Raises:
            ValueError: If an unsupported operation is specified.
        """
        if operation == 'rotate':
            theta = args[0]
            return cls._parallel_rotate_shapes(shapes, theta)
        elif operation == 'translate':
            dx, dy = args[0], args[1]
            return cls._parallel_translate_shapes(shapes, dx, dy)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    @classmethod
    def _parallel_rotate_shapes(cls, shapes: List[Shape], theta: float) -> List[Shape]:
        """Rotate multiple shapes in parallel.

        Args:
            shapes: List of shapes to rotate.
            theta: Rotation angle in radians.

        Returns:
            List of rotated Shape objects.
        """
        processed_shapes = []

        for shape in shapes:
            # Create a copy to avoid modifying original
            new_shape = cls.copy(shape)
            new_shape.rotate_around_center(theta)
            processed_shapes.append(new_shape)

        return processed_shapes

    @classmethod
    def _parallel_translate_shapes(
        cls,
        shapes: List[Shape],
        dx: float,
        dy: float
    ) -> List[Shape]:
        """Translate multiple shapes in parallel.

        Args:
            shapes: List of shapes to translate.
            dx: Translation amount in x direction.
            dy: Translation amount in y direction.

        Returns:
            List of translated Shape objects.
        """
        processed_shapes = []

        for shape in shapes:
            # Create a copy to avoid modifying original
            new_shape = cls.copy(shape)
            new_shape.translate_x(dx)
            new_shape.translate_y(dy)
            new_shape.centroid()  # Recalculate centroid after translation
            processed_shapes.append(new_shape)

        return processed_shapes

    @classmethod
    def batch_process_with_vmap(
        cls,
        shapes_points_list: List[Shape],
        operation_func: Callable[[jnp.ndarray], jnp.ndarray]
    ) -> jnp.ndarray:
        """Use JAX vmap for true parallel processing of shape operations.

        Args:
            shapes_points_list: List of Shape objects to process.
            operation_func: JAX-compatible function to apply to each shape's
                points array.

        Returns:
            Processed point arrays as a batched JAX array.
        """
        # Convert to JAX arrays
        points_arrays = [jnp.array([[p.x, p.y] for p in shape.points]) for shape in shapes_points_list]
        max_points = max(len(arr) for arr in points_arrays)

        # Pad arrays to same length for vmap
        padded_arrays = []
        for arr in points_arrays:
            if len(arr) < max_points:
                padding = jnp.zeros((max_points - len(arr), 2))
                arr = jnp.concatenate([arr, padding])
            padded_arrays.append(arr)

        batch_array = jnp.stack(padded_arrays)

        # Apply operation using vmap
        vectorized_op = vmap(operation_func)
        result = vectorized_op(batch_array)

        return result
