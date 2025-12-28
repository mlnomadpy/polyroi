"""Unit tests for the Shape class."""

import pytest
import math
import numpy as np
import jax.numpy as jnp
from polyroi import Shape, Point


class TestShapeInit:
    """Tests for Shape initialization."""

    def test_init_basic(self):
        """Test basic Shape initialization."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        assert len(shape.points) == 4

    def test_init_creates_point_objects(self):
        """Test that initialization creates Point objects."""
        points = [(5, 5), (15, 5), (15, 15)]
        shape = Shape(points)
        for p in shape.points:
            assert isinstance(p, Point)

    def test_init_calculates_centroid(self):
        """Test that initialization calculates centroid."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        assert hasattr(shape, 'center')
        assert isinstance(shape.center, Point)


class TestShapeCentroid:
    """Tests for Shape centroid calculation."""

    def test_centroid_square(self):
        """Test centroid of a square."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        center = shape.center.to_tuple()
        assert center[0] == pytest.approx(5.0, abs=1e-5)
        assert center[1] == pytest.approx(5.0, abs=1e-5)

    def test_centroid_rectangle(self):
        """Test centroid of a rectangle."""
        points = [(0, 0), (20, 0), (20, 10), (0, 10)]
        shape = Shape(points)
        center = shape.center.to_tuple()
        assert center[0] == pytest.approx(10.0, abs=1e-5)
        assert center[1] == pytest.approx(5.0, abs=1e-5)

    def test_centroid_triangle(self):
        """Test centroid of a triangle."""
        points = [(0, 0), (6, 0), (3, 6)]
        shape = Shape(points)
        center = shape.center.to_tuple()
        assert center[0] == pytest.approx(3.0, abs=1e-5)
        assert center[1] == pytest.approx(2.0, abs=1e-5)


class TestShapeTranslation:
    """Tests for Shape translation operations."""

    def test_translate_x_positive(self):
        """Test positive X translation."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        shape.translate_x(5)
        expected_x = [5.0, 15.0, 15.0, 5.0]
        for i, p in enumerate(shape.points):
            assert p.x == pytest.approx(expected_x[i], abs=1e-5)

    def test_translate_x_negative(self):
        """Test negative X translation."""
        points = [(10, 10), (20, 10), (20, 20), (10, 20)]
        shape = Shape(points)
        shape.translate_x(-5)
        expected_x = [5.0, 15.0, 15.0, 5.0]
        for i, p in enumerate(shape.points):
            assert p.x == pytest.approx(expected_x[i], abs=1e-5)

    def test_translate_y_positive(self):
        """Test positive Y translation."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        shape.translate_y(7)
        expected_y = [7.0, 7.0, 17.0, 17.0]
        for i, p in enumerate(shape.points):
            assert p.y == pytest.approx(expected_y[i], abs=1e-5)

    def test_translate_y_negative(self):
        """Test negative Y translation."""
        points = [(0, 10), (10, 10), (10, 20), (0, 20)]
        shape = Shape(points)
        shape.translate_y(-5)
        expected_y = [5.0, 5.0, 15.0, 15.0]
        for i, p in enumerate(shape.points):
            assert p.y == pytest.approx(expected_y[i], abs=1e-5)


class TestShapeTranslateTo:
    """Tests for Shape translate_to operation."""

    def test_translate_to_basic(self):
        """Test translate_to moves shape correctly."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        shape.translate_to(20, 20)
        # After translate_to, max_x should be 20 and max_y should be 20
        shape.to_rectangle()
        assert float(shape.max_x) == pytest.approx(20.0, abs=1e-5)
        assert float(shape.max_y) == pytest.approx(20.0, abs=1e-5)


class TestShapeRotation:
    """Tests for Shape rotation operations."""

    def test_rotate_around_center_preserves_centroid(self):
        """Test that rotation preserves centroid location."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        original_center = shape.center.to_tuple()
        shape.rotate_around_center(jnp.pi / 4)
        new_center = shape.center.to_tuple()
        assert new_center[0] == pytest.approx(original_center[0], abs=1e-5)
        assert new_center[1] == pytest.approx(original_center[1], abs=1e-5)

    def test_rotate_360_degrees(self):
        """Test 360 degree rotation returns to original."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        original_points = [p.to_tuple() for p in shape.points]
        shape.rotate_around_center(2 * jnp.pi)
        for i, p in enumerate(shape.points):
            assert p.x == pytest.approx(original_points[i][0], abs=1e-5)
            assert p.y == pytest.approx(original_points[i][1], abs=1e-5)


class TestShapeToRectangle:
    """Tests for Shape to_rectangle bounding box calculation."""

    def test_to_rectangle_square(self):
        """Test bounding box for square."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        bbox = shape.to_rectangle()
        assert bbox == [(0.0, 0.0), (10.0, 10.0)]

    def test_to_rectangle_sets_dimensions(self):
        """Test that to_rectangle sets width and height."""
        points = [(5, 10), (25, 10), (25, 30), (5, 30)]
        shape = Shape(points)
        shape.to_rectangle()
        assert float(shape.width) == pytest.approx(20.0, abs=1e-5)
        assert float(shape.height) == pytest.approx(20.0, abs=1e-5)

    def test_to_rectangle_irregular(self):
        """Test bounding box for irregular shape."""
        points = [(5, 5), (15, 2), (20, 15), (10, 20), (0, 12)]
        shape = Shape(points)
        bbox = shape.to_rectangle()
        assert bbox[0][0] == pytest.approx(0.0, abs=1e-5)  # min_x
        assert bbox[0][1] == pytest.approx(2.0, abs=1e-5)  # min_y
        assert bbox[1][0] == pytest.approx(20.0, abs=1e-5)  # max_x
        assert bbox[1][1] == pytest.approx(20.0, abs=1e-5)  # max_y


class TestShapeToArray:
    """Tests for Shape to_array conversion."""

    def test_to_array_basic(self):
        """Test basic to_array conversion."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        arr = shape.to_array()
        assert arr.shape == (4, 2)

    def test_to_array_values(self):
        """Test that to_array preserves values."""
        points = [(5, 10), (15, 10), (15, 20), (5, 20)]
        shape = Shape(points)
        arr = shape.to_array()
        expected = jnp.array([[5, 10], [15, 10], [15, 20], [5, 20]])
        np.testing.assert_array_almost_equal(np.array(arr), np.array(expected))


class TestShapeCopy:
    """Tests for Shape copy operation."""

    def test_copy_creates_new_shape(self):
        """Test that copy creates a new Shape object."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        copied = Shape.copy(shape)
        assert copied is not shape

    def test_copy_has_same_points(self):
        """Test that copy has the same points."""
        points = [(5, 5), (15, 5), (15, 15), (5, 15)]
        shape = Shape(points)
        copied = Shape.copy(shape)
        for i, p in enumerate(copied.points):
            assert p.to_tuple() == shape.points[i].to_tuple()

    def test_copy_is_independent(self):
        """Test that copy is independent of original."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        copied = Shape.copy(shape)
        copied.translate_x(100)
        # Original should be unchanged
        assert shape.points[0].x == pytest.approx(0.0, abs=1e-5)


class TestShapeUpdate:
    """Tests for Shape update operation."""

    def test_update_applies_transformations(self):
        """Test that update applies rotation and translation."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shape = Shape(points)
        original_points = [(p.x, p.y) for p in shape.points]
        shape.update(5, 5, 0)  # Just translation
        # Verify points are translated
        for i, p in enumerate(shape.points):
            assert p.x == pytest.approx(original_points[i][0] + 5, abs=1e-5)
            assert p.y == pytest.approx(original_points[i][1] + 5, abs=1e-5)


class TestShapeStr:
    """Tests for Shape string representation."""

    def test_str_contains_points(self):
        """Test that string representation contains point info."""
        points = [(0, 0), (10, 0)]
        shape = Shape(points)
        result = str(shape)
        assert "Printing the shapes points" in result


class TestShapeExtractContent:
    """Tests for Shape extract_content operation."""

    def test_extract_content_returns_array(self):
        """Test that extract_content returns numpy array."""
        points = [(10, 10), (50, 10), (50, 50), (10, 50)]
        shape = Shape(points)
        # Create a simple test image
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = shape.extract_content(frame)
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape


class TestShapeGetHistogram:
    """Tests for Shape histogram calculation."""

    def test_get_histogram_shape(self):
        """Test that histogram has correct shape."""
        points = [(10, 10), (50, 10), (50, 50), (10, 50)]
        shape = Shape(points)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        histogram = shape.get_histogram(frame)
        assert histogram.shape == (3, 255)


class TestShapeParallelProcessing:
    """Tests for Shape parallel processing operations."""

    def test_process_multiple_shapes_rotate(self):
        """Test parallel rotation of multiple shapes."""
        shapes = [
            Shape([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Shape([(20, 20), (30, 20), (30, 30), (20, 30)])
        ]
        rotated = Shape.process_multiple_shapes_parallel(shapes, 'rotate', jnp.pi / 4)
        assert len(rotated) == 2
        # Check that original shapes are unchanged
        assert shapes[0].points[0].x == pytest.approx(0.0, abs=1e-5)

    def test_process_multiple_shapes_translate(self):
        """Test parallel translation of multiple shapes."""
        shapes = [
            Shape([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Shape([(20, 20), (30, 20), (30, 30), (20, 30)])
        ]
        translated = Shape.process_multiple_shapes_parallel(shapes, 'translate', 5, 5)
        assert len(translated) == 2
        # First shape should have moved
        assert translated[0].points[0].x == pytest.approx(5.0, abs=1e-5)
        assert translated[0].points[0].y == pytest.approx(5.0, abs=1e-5)

    def test_process_multiple_shapes_invalid_operation(self):
        """Test that invalid operation raises error."""
        shapes = [Shape([(0, 0), (10, 0), (10, 10), (0, 10)])]
        with pytest.raises(ValueError, match="Unsupported operation"):
            Shape.process_multiple_shapes_parallel(shapes, 'invalid_op')


class TestShapeBatchProcessWithVmap:
    """Tests for batch processing with vmap."""

    def test_batch_process_returns_result(self):
        """Test that batch_process_with_vmap returns results."""
        shapes = [
            Shape([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Shape([(5, 5), (15, 5), (15, 15), (5, 15)])
        ]
        
        def identity_op(points):
            return points
        
        result = Shape.batch_process_with_vmap(shapes, identity_op)
        assert result.shape[0] == 2  # Two shapes
