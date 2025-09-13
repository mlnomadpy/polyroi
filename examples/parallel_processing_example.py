#!/usr/bin/env python3
"""
Parallel processing example with JAX vmap

This example demonstrates how to use JAX's vectorization capabilities
to process multiple ROI shapes in parallel, significantly improving
performance for batch operations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax.numpy as jnp
from jax import vmap
import time
from polyroi import Shape, Point

def create_multiple_shapes(n_shapes=10):
    """Create multiple test shapes"""
    shapes = []
    for i in range(n_shapes):
        # Create different sized rectangles at different positions
        base_x, base_y = i * 50, i * 30
        size = 20 + i * 5
        points = [
            (base_x, base_y),
            (base_x + size, base_y),
            (base_x + size, base_y + size),
            (base_x, base_y + size)
        ]
        shapes.append(Shape(points))
    return shapes

def demonstrate_parallel_rotation():
    """Demonstrate parallel rotation of multiple shapes"""
    print("=== Parallel Shape Rotation ===\n")
    
    # Create test shapes
    shapes = create_multiple_shapes(5)
    print(f"Created {len(shapes)} shapes")
    
    # Show original centroids
    print("Original centroids:")
    for i, shape in enumerate(shapes):
        print(f"  Shape {i}: {shape.center.to_tuple()}")
    
    # Rotate all shapes in parallel
    rotation_angle = jnp.pi / 6  # 30 degrees
    start_time = time.time()
    rotated_shapes = Shape.process_multiple_shapes_parallel(shapes, 'rotate', rotation_angle)
    parallel_time = time.time() - start_time
    
    print(f"\nAfter parallel rotation ({rotation_angle:.3f} radians):")
    for i, shape in enumerate(rotated_shapes):
        print(f"  Shape {i}: {shape.center.to_tuple()}")
    
    print(f"Parallel processing time: {parallel_time:.6f} seconds")

def demonstrate_parallel_translation():
    """Demonstrate parallel translation of multiple shapes"""
    print("\n=== Parallel Shape Translation ===\n")
    
    # Create test shapes
    shapes = create_multiple_shapes(5)
    
    # Translate all shapes in parallel
    dx, dy = 100, 50
    start_time = time.time()
    translated_shapes = Shape.process_multiple_shapes_parallel(shapes, 'translate', dx, dy)
    parallel_time = time.time() - start_time
    
    print(f"Translation by ({dx}, {dy}):")
    print("Original vs Translated centroids:")
    for i, (orig, trans) in enumerate(zip(shapes, translated_shapes)):
        orig_center = orig.center.to_tuple()
        trans_center = trans.center.to_tuple()
        print(f"  Shape {i}: {orig_center} -> {trans_center}")
    
    print(f"Parallel processing time: {parallel_time:.6f} seconds")

def demonstrate_vectorized_operations():
    """Demonstrate JAX vectorized operations on shape data"""
    print("\n=== JAX Vectorized Operations ===\n")
    
    # Create shapes and convert to arrays
    shapes = create_multiple_shapes(3)
    
    print("Demonstrating vectorized centroid calculation:")
    for i, shape in enumerate(shapes):
        points_array = shape.to_array()
        print(f"Shape {i} points:\n{points_array}")
        
        # Vectorized centroid calculation using JAX
        centroid = jnp.mean(points_array, axis=0)
        print(f"JAX vectorized centroid: {centroid}")
        print(f"Shape's centroid method: {shape.center.to_tuple()}")
        print()

def performance_comparison():
    """Compare sequential vs parallel processing performance"""
    print("=== Performance Comparison ===\n")
    
    # Create larger number of shapes for meaningful comparison
    n_shapes = 100
    shapes = create_multiple_shapes(n_shapes)
    print(f"Testing with {n_shapes} shapes")
    
    # Sequential processing
    start_time = time.time()
    sequential_results = []
    for shape in shapes:
        new_shape = Shape.copy(shape)
        new_shape.rotate_around_center(0.1)
        new_shape.translate_x(10)
        sequential_results.append(new_shape)
    sequential_time = time.time() - start_time
    
    # Parallel processing (rotation then translation)
    start_time = time.time()
    parallel_rotated = Shape.process_multiple_shapes_parallel(shapes, 'rotate', 0.1)
    parallel_results = Shape.process_multiple_shapes_parallel(parallel_rotated, 'translate', 10, 0)
    parallel_time = time.time() - start_time
    
    print(f"Sequential processing time: {sequential_time:.6f} seconds")
    print(f"Parallel processing time: {parallel_time:.6f} seconds")
    print(f"Speedup factor: {sequential_time / parallel_time:.2f}x")
    
    # Verify results are similar
    for i in range(min(3, len(shapes))):
        seq_center = sequential_results[i].center.to_tuple()
        par_center = parallel_results[i].center.to_tuple()
        print(f"Shape {i} - Sequential: {seq_center}, Parallel: {par_center}")

def demonstrate_jax_vmap_potential():
    """Show how JAX vmap could be used for true vectorization"""
    print("\n=== JAX vmap Potential ===\n")
    
    # Create sample point arrays
    points_arrays = [
        jnp.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        jnp.array([[20, 20], [30, 20], [30, 30], [20, 30]]),
        jnp.array([[40, 40], [50, 40], [50, 50], [40, 50]])
    ]
    
    # Define a simple transformation function
    def rotate_points(points, angle=jnp.pi/4):
        """Rotate points around their centroid"""
        centroid = jnp.mean(points, axis=0)
        centered = points - centroid
        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = jnp.dot(centered, rotation_matrix.T)
        return rotated + centroid
    
    # Apply transformation using vmap for parallel processing
    batch_points = jnp.stack(points_arrays)
    print(f"Batch input shape: {batch_points.shape}")
    
    # Vectorized rotation using vmap
    vectorized_rotate = vmap(rotate_points)
    rotated_batch = vectorized_rotate(batch_points)
    
    print(f"Batch output shape: {rotated_batch.shape}")
    print("Original vs Rotated points:")
    for i, (orig, rot) in enumerate(zip(batch_points, rotated_batch)):
        print(f"  Set {i}:")
        print(f"    Original centroid: {jnp.mean(orig, axis=0)}")
        print(f"    Rotated centroid:  {jnp.mean(rot, axis=0)}")

if __name__ == "__main__":
    demonstrate_parallel_rotation()
    demonstrate_parallel_translation()
    demonstrate_vectorized_operations()
    performance_comparison()
    demonstrate_jax_vmap_potential()
    print("\n=== Parallel processing example completed! ===")