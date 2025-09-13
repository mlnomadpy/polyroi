# PolyROI JAX Examples

This directory contains examples demonstrating the JAX-enhanced capabilities of the polyroi library.

## Examples Overview

### 1. Basic JAX Example (`basic_jax_example.py`)
Demonstrates the fundamental operations with JAX numpy backend:
- Creating and manipulating shapes
- Point operations with JAX functions
- Basic performance comparisons
- JAX array conversions

**Run with:**
```bash
python examples/basic_jax_example.py
```

### 2. Parallel Processing Example (`parallel_processing_example.py`)
Shows how to use JAX's vectorization for batch operations:
- Parallel rotation of multiple shapes
- Parallel translation operations
- Performance comparisons between sequential and parallel processing
- JAX vmap demonstrations

**Run with:**
```bash
python examples/parallel_processing_example.py
```

### 3. Image Processing Example (`image_processing_example.py`)
Demonstrates real-world image processing scenarios:
- ROI extraction from synthetic images
- Batch processing of multiple ROIs
- Advanced JAX operations on shape data
- Visualization creation
- Performance analysis with image data

**Run with:**
```bash
python examples/image_processing_example.py
```

## Key JAX Features Demonstrated

### 1. **Automatic Differentiation Ready**
All mathematical operations now use JAX numpy, making the library compatible with gradient computation.

### 2. **Vectorized Operations**
- `jax.vmap` for parallel processing of multiple shapes
- Vectorized mathematical operations on point arrays
- Batch transformations

### 3. **JIT Compilation Potential**
While not explicitly demonstrated (to maintain compatibility), all operations are JAX-compatible and can be JIT compiled for additional performance.

### 4. **GPU Acceleration Ready**
JAX operations can automatically utilize GPU acceleration when available.

## Performance Benefits

The JAX backend provides several advantages:

1. **Parallel Processing**: Process multiple shapes simultaneously
2. **Vectorized Operations**: Efficient operations on arrays of points
3. **Memory Efficiency**: Better memory management for large datasets
4. **Future-Proof**: Compatible with modern ML workflows

## API Compatibility

The JAX migration maintains full API compatibility with the original library:
- All existing methods work exactly the same way
- No breaking changes to the public interface
- Additional parallel processing methods are added as new functionality

## Requirements

```
numpy
opencv-python
jax
jaxlib
```

## Usage Tips

1. **For single shapes**: Use the existing API methods (work identically)
2. **For multiple shapes**: Use the new `process_multiple_shapes_parallel()` methods
3. **For custom operations**: Leverage `to_array()` to get JAX arrays for advanced operations
4. **For GPU acceleration**: Ensure JAX is configured for your GPU setup

## Example Output

Running the examples will show:
- Performance comparisons between sequential and parallel processing
- Demonstration of vectorized operations
- Visual outputs (saved images in the examples directory)
- Timing information for different operation types