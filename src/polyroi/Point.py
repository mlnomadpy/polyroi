"""Point class for representing 2D points with JAX-accelerated operations."""

from __future__ import annotations
from typing import Tuple

import jax.numpy as jnp
import cv2 as cv
import numpy as np


class Point:
    """A 2D point with coordinates (x, y) and various transformation methods.

    This class provides methods for distance calculation, translation, rotation,
    and coordinate conversion. All mathematical operations use JAX numpy for
    potential GPU acceleration.

    Attributes:
        x: The x-coordinate (rounded to nearest integer).
        y: The y-coordinate (rounded to nearest integer).
    """

    def __init__(self, x: float, y: float) -> None:
        """Initialize a Point with x and y coordinates.

        Args:
            x: The x-coordinate.
            y: The y-coordinate.
        """
        self.x = x
        self.y = y
        self.round_point()

    def distance(self, point: Point) -> jnp.ndarray:
        """Calculate Euclidean distance to another point.

        Args:
            point: The other point to calculate distance to.

        Returns:
            The Euclidean distance between this point and the other point.
        """
        return jnp.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)

    def translate_x(self, x: float) -> None:
        """Translate the point along the x-axis.

        Args:
            x: The amount to translate in the x direction.
        """
        self.x += x
        self.round_point()

    def translate_y(self, y: float) -> None:
        """Translate the point along the y-axis.

        Args:
            y: The amount to translate in the y direction.
        """
        self.y += y
        self.round_point()

    def update(self, x: float, y: float, theta: float) -> None:
        """Apply translation and rotation to the point.

        Args:
            x: The amount to translate in the x direction.
            y: The amount to translate in the y direction.
            theta: The angle to rotate in radians.
        """
        self.translate_x(x)
        self.translate_y(y)
        self.rotate(theta)

    def rotate(self, theta: float) -> None:
        """Rotate the point around the origin by the given angle.

        Args:
            theta: The angle to rotate in radians.
        """
        old_x = self.x
        self.x = self.x * jnp.cos(theta) - self.y * jnp.sin(theta)
        self.y = old_x * jnp.sin(theta) + self.y * jnp.cos(theta)
        self.round_point()

    def to_cylindrical(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Convert Cartesian coordinates to cylindrical (polar) coordinates.

        Returns:
            A tuple (r, theta) where r is the radius and theta is the angle.
        """
        r = jnp.sqrt(self.x**2 + self.y**2)
        theta = jnp.arctan(self.y/self.x)
        return (r, theta)

    def from_cylindrical(self, r: float, theta: float) -> None:
        """Set point coordinates from cylindrical (polar) coordinates.

        Args:
            r: The radius (distance from origin).
            theta: The angle in radians.
        """
        self.x = r * jnp.cos(theta)
        self.y = r * jnp.sin(theta)
        self.round_point()

    def round_point(self) -> None:
        """Round the point coordinates to the nearest integer."""
        self.x = float(jnp.round(self.x))
        self.y = float(jnp.round(self.y))

    def to_tuple(self) -> Tuple[float, float]:
        """Convert the point to a tuple.

        Returns:
            A tuple (x, y) of the point coordinates.
        """
        return (self.x, self.y)

    def draw_point(self, frame: np.ndarray) -> None:
        """Draw the point on an image frame.

        Args:
            frame: The image frame (numpy array) to draw on.
        """
        # Convert to integers for OpenCV
        point_int = (int(self.x), int(self.y))
        cv.circle(frame, point_int, radius=1,
                  color=(255, 0, 255), thickness=1)

    def __str__(self) -> str:
        """Return a string representation of the point.

        Returns:
            A string showing the x and y coordinates.
        """
        return "X: {} Y: {} \n".format(self.x, self.y)

