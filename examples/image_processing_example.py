#!/usr/bin/env python3
"""
Interactive image processing example with JAX-enabled polyroi

This example demonstrates how to use the JAX-enabled polyroi library
for interactive image processing, including ROI extraction and batch
processing of multiple regions.

Note: This example creates synthetic images since GUI interaction
may not be available in all environments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import jax.numpy as jnp
import cv2 as cv
from polyroi import Shape, Point

def create_synthetic_image(width=400, height=300):
    """Create a synthetic test image with different regions"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add colored rectangles
    cv.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
    cv.rectangle(img, (200, 50), (300, 150), (0, 255, 0), -1)  # Green rectangle
    cv.rectangle(img, (50, 180), (150, 280), (0, 0, 255), -1)  # Red rectangle
    cv.rectangle(img, (200, 180), (300, 280), (255, 255, 0), -1)  # Cyan rectangle
    
    # Add some noise
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    img = cv.add(img, noise)
    
    return img

def create_predefined_rois():
    """Create predefined ROI shapes for testing"""
    rois = []
    
    # ROI for blue rectangle
    roi1_points = [(60, 60), (140, 60), (140, 140), (60, 140)]
    rois.append(Shape(roi1_points))
    
    # ROI for green rectangle  
    roi2_points = [(210, 60), (290, 60), (290, 140), (210, 140)]
    rois.append(Shape(roi2_points))
    
    # ROI for red rectangle
    roi3_points = [(60, 190), (140, 190), (140, 270), (60, 270)]
    rois.append(Shape(roi3_points))
    
    # Irregular ROI for cyan rectangle
    roi4_points = [(210, 190), (290, 200), (280, 270), (220, 260)]
    rois.append(Shape(roi4_points))
    
    return rois

def demonstrate_roi_extraction():
    """Demonstrate ROI extraction with multiple shapes"""
    print("=== ROI Extraction with JAX Backend ===\n")
    
    # Create test image and ROIs
    img = create_synthetic_image()
    rois = create_predefined_rois()
    
    print(f"Created image with shape: {img.shape}")
    print(f"Created {len(rois)} ROI shapes")
    
    # Extract content from each ROI
    for i, roi in enumerate(rois):
        print(f"\nROI {i+1}:")
        print(f"  Points: {[p.to_tuple() for p in roi.points]}")
        print(f"  Centroid: {roi.center.to_tuple()}")
        
        # Extract ROI content
        roi_content = roi.extract_content(img)
        
        # Calculate bounding box
        bbox = roi.to_rectangle()
        print(f"  Bounding box: {bbox}")
        
        # Get histogram
        histogram = roi.get_histogram(img)
        print(f"  Histogram shape: {histogram.shape}")
        print(f"  Mean RGB values: {np.mean(histogram, axis=1)}")

def demonstrate_batch_roi_processing():
    """Demonstrate batch processing of multiple ROIs"""
    print("\n=== Batch ROI Processing ===\n")
    
    # Create test data
    img = create_synthetic_image()
    rois = create_predefined_rois()
    
    # Demonstrate parallel transformation of ROIs
    print("Original ROI centroids:")
    for i, roi in enumerate(rois):
        print(f"  ROI {i+1}: {roi.center.to_tuple()}")
    
    # Apply parallel rotation
    rotation_angle = jnp.pi / 8  # 22.5 degrees
    rotated_rois = Shape.process_multiple_shapes_parallel(rois, 'rotate', rotation_angle)
    
    print(f"\nAfter parallel rotation by {rotation_angle:.3f} radians:")
    for i, roi in enumerate(rotated_rois):
        print(f"  ROI {i+1}: {roi.center.to_tuple()}")
    
    # Apply parallel translation
    dx, dy = 20, -10
    translated_rois = Shape.process_multiple_shapes_parallel(rotated_rois, 'translate', dx, dy)
    
    print(f"\nAfter parallel translation by ({dx}, {dy}):")
    for i, roi in enumerate(translated_rois):
        print(f"  ROI {i+1}: {roi.center.to_tuple()}")

def demonstrate_advanced_jax_operations():
    """Demonstrate advanced JAX operations on ROI data"""
    print("\n=== Advanced JAX Operations ===\n")
    
    # Create test ROIs
    rois = create_predefined_rois()
    
    # Convert all ROI points to JAX arrays
    roi_arrays = [roi.to_array() for roi in rois]
    
    print("JAX array operations on ROI data:")
    for i, roi_array in enumerate(roi_arrays):
        print(f"\nROI {i+1}:")
        print(f"  Shape: {roi_array.shape}")
        print(f"  Points:\n{roi_array}")
        
        # JAX operations
        centroid = jnp.mean(roi_array, axis=0)
        distances_from_centroid = jnp.linalg.norm(roi_array - centroid, axis=1)
        area_approx = jnp.prod(jnp.max(roi_array, axis=0) - jnp.min(roi_array, axis=0))
        
        print(f"  JAX centroid: {centroid}")
        print(f"  Distances from centroid: {distances_from_centroid}")
        print(f"  Approximate area: {area_approx}")

def save_visualization():
    """Save a visualization of the ROI processing"""
    print("\n=== Creating Visualization ===\n")
    
    # Create test image and ROIs
    img = create_synthetic_image()
    rois = create_predefined_rois()
    
    # Create visualization image
    vis_img = img.copy()
    
    # Draw original ROIs in yellow
    for roi in rois:
        roi.draw_shape(vis_img, color=(0, 255, 255), thickness=2)
    
    # Create transformed ROIs and draw in cyan
    transformed_rois = Shape.process_multiple_shapes_parallel(rois, 'rotate', jnp.pi/6)
    transformed_rois = Shape.process_multiple_shapes_parallel(transformed_rois, 'translate', 30, 20)
    
    for roi in transformed_rois:
        roi.draw_shape(vis_img, color=(255, 255, 0), thickness=2)
    
    # Save the visualization
    output_path = os.path.join(os.path.dirname(__file__), 'roi_visualization.jpg')
    cv.imwrite(output_path, vis_img)
    print(f"Visualization saved to: {output_path}")
    
    return vis_img

def demonstrate_performance_with_images():
    """Demonstrate performance improvements with image processing"""
    print("\n=== Performance with Image Processing ===\n")
    
    import time
    
    # Create larger test case
    img = create_synthetic_image(800, 600)
    
    # Create many ROIs
    rois = []
    for i in range(20):
        x, y = i * 30 + 10, (i % 10) * 50 + 10
        size = 40
        points = [(x, y), (x+size, y), (x+size, y+size), (x, y+size)]
        rois.append(Shape(points))
    
    print(f"Processing {len(rois)} ROIs on {img.shape} image")
    
    # Sequential processing
    start_time = time.time()
    sequential_histograms = []
    for roi in rois:
        hist = roi.get_histogram(img)
        sequential_histograms.append(hist)
    sequential_time = time.time() - start_time
    
    print(f"Sequential histogram calculation: {sequential_time:.4f} seconds")
    
    # The parallel processing is mainly for geometric transformations
    # Histogram calculation still needs to be done sequentially due to OpenCV
    start_time = time.time()
    rotated_rois = Shape.process_multiple_shapes_parallel(rois, 'rotate', 0.1)
    parallel_histograms = []
    for roi in rotated_rois:
        hist = roi.get_histogram(img)
        parallel_histograms.append(hist)
    mixed_time = time.time() - start_time
    
    print(f"Parallel transformation + sequential histogram: {mixed_time:.4f} seconds")
    print("Note: Full parallelization requires JAX-compatible image operations")

if __name__ == "__main__":
    demonstrate_roi_extraction()
    demonstrate_batch_roi_processing()
    demonstrate_advanced_jax_operations()
    save_visualization()
    demonstrate_performance_with_images()
    print("\n=== Image processing example completed! ===")