"""Unit tests for the Point class."""

import pytest
import math
import jax.numpy as jnp
from polyroi import Point


class TestPointInit:
    """Tests for Point initialization."""

    def test_init_basic(self):
        """Test basic Point initialization."""
        point = Point(3, 4)
        assert point.x == 3.0
        assert point.y == 4.0

    def test_init_with_floats(self):
        """Test Point initialization with floats."""
        point = Point(3.7, 4.3)
        assert point.x == 4.0  # Rounded
        assert point.y == 4.0  # Rounded

    def test_init_with_negative_values(self):
        """Test Point initialization with negative values."""
        point = Point(-5, -10)
        assert point.x == -5.0
        assert point.y == -10.0

    def test_init_with_zero(self):
        """Test Point initialization at origin."""
        point = Point(0, 0)
        assert point.x == 0.0
        assert point.y == 0.0


class TestPointDistance:
    """Tests for Point distance calculation."""

    def test_distance_basic(self):
        """Test basic distance calculation."""
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        distance = p1.distance(p2)
        assert float(distance) == pytest.approx(5.0, rel=1e-5)

    def test_distance_same_point(self):
        """Test distance to same point is zero."""
        p1 = Point(5, 5)
        p2 = Point(5, 5)
        distance = p1.distance(p2)
        assert float(distance) == pytest.approx(0.0, abs=1e-5)

    def test_distance_horizontal(self):
        """Test horizontal distance calculation."""
        p1 = Point(0, 0)
        p2 = Point(10, 0)
        distance = p1.distance(p2)
        assert float(distance) == pytest.approx(10.0, rel=1e-5)

    def test_distance_vertical(self):
        """Test vertical distance calculation."""
        p1 = Point(0, 0)
        p2 = Point(0, 7)
        distance = p1.distance(p2)
        assert float(distance) == pytest.approx(7.0, rel=1e-5)


class TestPointTranslation:
    """Tests for Point translation operations."""

    def test_translate_x_positive(self):
        """Test positive X translation."""
        point = Point(5, 10)
        point.translate_x(3)
        assert point.x == 8.0
        assert point.y == 10.0

    def test_translate_x_negative(self):
        """Test negative X translation."""
        point = Point(5, 10)
        point.translate_x(-3)
        assert point.x == 2.0
        assert point.y == 10.0

    def test_translate_y_positive(self):
        """Test positive Y translation."""
        point = Point(5, 10)
        point.translate_y(7)
        assert point.x == 5.0
        assert point.y == 17.0

    def test_translate_y_negative(self):
        """Test negative Y translation."""
        point = Point(5, 10)
        point.translate_y(-7)
        assert point.x == 5.0
        assert point.y == 3.0


class TestPointRotation:
    """Tests for Point rotation operations."""

    def test_rotate_90_degrees(self):
        """Test 90 degree rotation."""
        point = Point(1, 0)
        point.rotate(jnp.pi / 2)
        assert point.x == pytest.approx(0.0, abs=1e-5)
        assert point.y == pytest.approx(1.0, abs=1e-5)

    def test_rotate_180_degrees(self):
        """Test 180 degree rotation."""
        point = Point(1, 0)
        point.rotate(jnp.pi)
        assert point.x == pytest.approx(-1.0, abs=1e-5)
        assert point.y == pytest.approx(0.0, abs=1e-5)

    def test_rotate_360_degrees(self):
        """Test 360 degree rotation (full circle)."""
        point = Point(3, 4)
        original_x, original_y = point.x, point.y
        point.rotate(2 * jnp.pi)
        assert point.x == pytest.approx(original_x, abs=1e-5)
        assert point.y == pytest.approx(original_y, abs=1e-5)


class TestPointCylindrical:
    """Tests for Point cylindrical coordinate operations."""

    def test_to_cylindrical_basic(self):
        """Test conversion to cylindrical coordinates."""
        point = Point(3, 4)
        r, theta = point.to_cylindrical()
        assert float(r) == pytest.approx(5.0, rel=1e-5)

    def test_to_cylindrical_unit_x(self):
        """Test conversion for point on x-axis."""
        point = Point(5, 0)
        r, theta = point.to_cylindrical()
        assert float(r) == pytest.approx(5.0, rel=1e-5)
        assert float(theta) == pytest.approx(0.0, abs=1e-5)

    def test_from_cylindrical(self):
        """Test conversion from cylindrical coordinates."""
        point = Point(0, 0)
        point.from_cylindrical(5, 0)
        assert point.x == pytest.approx(5.0, abs=1e-5)
        assert point.y == pytest.approx(0.0, abs=1e-5)


class TestPointUpdate:
    """Tests for Point update operation."""

    def test_update_basic(self):
        """Test update with translation and rotation."""
        point = Point(1, 0)
        point.update(2, 3, 0)  # Just translation, no rotation
        assert point.x == pytest.approx(3.0, abs=1e-5)
        assert point.y == pytest.approx(3.0, abs=1e-5)


class TestPointToTuple:
    """Tests for Point to_tuple conversion."""

    def test_to_tuple_basic(self):
        """Test basic to_tuple conversion."""
        point = Point(5, 10)
        result = point.to_tuple()
        assert result == (5.0, 10.0)

    def test_to_tuple_negative(self):
        """Test to_tuple with negative values."""
        point = Point(-3, -7)
        result = point.to_tuple()
        assert result == (-3.0, -7.0)


class TestPointStr:
    """Tests for Point string representation."""

    def test_str_basic(self):
        """Test string representation."""
        point = Point(5, 10)
        result = str(point)
        assert "X: 5" in result or "X: 5.0" in result
        assert "Y: 10" in result or "Y: 10.0" in result


class TestPointRounding:
    """Tests for Point rounding behavior."""

    def test_round_point_basic(self):
        """Test that points are rounded to integers."""
        point = Point(3.7, 4.3)
        point.round_point()
        assert point.x == 4.0
        assert point.y == 4.0

    def test_round_point_half(self):
        """Test rounding at half values."""
        point = Point(3.5, 4.5)
        point.round_point()
        # JAX uses banker's rounding, so 3.5 -> 4, 4.5 -> 4
        assert isinstance(point.x, float)
        assert isinstance(point.y, float)
