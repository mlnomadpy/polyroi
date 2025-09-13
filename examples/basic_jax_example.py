#!/usr/bin/env python3
"""
Basic JAX-enabled polyroi example

This example demonstrates the basic usage of polyroi with JAX numpy backend.
It creates a polygon shape, performs transformations, and shows how JAX
operations work seamlessly with the existing API.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax.numpy as jnp
from polyroi import Shape, Point

def create_sample_shape():
    """Create a sample rectangular shape"""
    points = [(100, 100), (200, 100), (200, 200), (100, 200)]
    return Shape(points)

def demonstrate_basic_operations():
    """Demonstrate basic operations with JAX backend"""
    print("=== Basic JAX-enabled polyroi Operations ===\n")
    
    # Create a shape
    shape = create_sample_shape()
    print(f"Original shape points: {[p.to_tuple() for p in shape.points]}")
    print(f"Centroid: {shape.center.to_tuple()}")
    
    # Create a copy for transformations
    transformed_shape = Shape.copy(shape)
    
    # Translate the shape
    print("\n--- Translation ---")
    transformed_shape.translate_x(50)
    transformed_shape.translate_y(30)
    print(f"After translation (50, 30): {[p.to_tuple() for p in transformed_shape.points]}")
    
    # Rotate the shape
    print("\n--- Rotation ---")
    transformed_shape.rotate_around_center(jnp.pi / 4)  # 45 degrees
    print(f"After 45° rotation: {[p.to_tuple() for p in transformed_shape.points]}")
    
    # Get bounding rectangle
    print("\n--- Bounding Rectangle ---")
    bbox = transformed_shape.to_rectangle()
    print(f"Bounding box: {bbox}")
    
    # Convert to array (now using JAX)
    print("\n--- JAX Array Conversion ---")
    points_array = transformed_shape.to_array()
    print(f"Points as JAX array:\n{points_array}")
    print(f"Array type: {type(points_array)}")

def demonstrate_point_operations():
    """Demonstrate Point class operations with JAX"""
    print("\n=== Point Operations with JAX ===\n")
    
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    
    print(f"Point 1: {p1.to_tuple()}")
    print(f"Point 2: {p2.to_tuple()}")
    
    # Distance calculation (using JAX sqrt)
    distance = p1.distance(p2)
    print(f"Distance between points: {distance}")
    
    # Cylindrical coordinates
    r, theta = p2.to_cylindrical()
    print(f"Point 2 in cylindrical: r={r}, theta={theta}")
    
    # Rotation
    p3 = Point(1, 0)
    print(f"Point 3 before rotation: {p3.to_tuple()}")
    p3.rotate(jnp.pi / 2)  # 90 degrees
    print(f"Point 3 after 90° rotation: {p3.to_tuple()}")

def compare_jax_vs_numpy():
    """Compare JAX vs NumPy performance for operations"""
    print("\n=== JAX vs NumPy Comparison ===\n")
    
    # Create test data
    points = [(i, i*2) for i in range(100)]
    shape = Shape(points)
    
    import time
    
    # Time JAX operations
    start_time = time.time()
    for _ in range(100):
        test_shape = Shape.copy(shape)
        test_shape.rotate_around_center(0.1)
    jax_time = time.time() - start_time
    
    print(f"100 rotations with JAX backend: {jax_time:.4f} seconds")
    print("Note: JAX operations are optimized for GPU acceleration and larger datasets")

if __name__ == "__main__":
    demonstrate_basic_operations()
    demonstrate_point_operations()
    compare_jax_vs_numpy()
    print("\n=== Example completed successfully! ===")