# TokenVector.Vision Release Notes - v1.0.0

**Release Date:** September 2026  
**Build Target:** .NET 8.0 / 9.0 Native AOT, C# 12.0

---

## What's New in v1.0.0

### Core Unmanaged Memory Engine (`Common/`)
* **`ImageBuffer`**: 64-byte aligned unmanaged buffer (`byte*`, `float*`) supporting Zero-GC life cycle, reference counting, and zero-copy 2D sub-region slicing.
* **Layouts**: First-class support for `HWC`, `CHW`, `NCHW`, and `NHWC`.

### Zero-Copy Numerics Bridge (`Interop/`)
* `ToNDArray` & `AsImageBuffer`: Seamless bidirectional tensor conversion with `TokenVector.Numerics.Core.NDArray<float>` / `NDArray<byte>`.

### Hardware-Accelerated Transforms (`Transforms/`)
* `ResizeTransform`: AVX2 Bilinear and Bicubic interpolation kernels.
* `CropTransforms`: `CenterCrop`, `RandomCrop`, and `PadTransform`.
* `GeometricTransforms`: `HorizontalFlip`, `VerticalFlip`, and continuous `Rotate` (Affine Matrix).
* `PhotometricTransforms`: `ColorJitter` (Brightness, Contrast, Saturation, Hue), `AdjustGamma`, and `GrayscaleTransform`.
* `NormalizeTransform`: Vectorized $(x - \text{mean}) / \text{std}$.
* `LayoutTransforms`: SIMD Transpose $HWC \leftrightarrow CHW$.
* **`FusedTransforms.ResizeNormalizeCHW`**: 1-pass fused operator delivering 3.24x speedup and 0 bytes GC allocation.

### Spatial Filters & Edge Detection (`Filters/`)
* `GaussianBlur`: Separable 1D Horizontal + Vertical AVX2 filter.
* `Convolution2D`: $3\times 3, 5\times 5, 7\times 7$ spatial convolution engine.
* `EdgeDetection`: Sobel X/Y, Magnitude, Laplacian.
* `CannyDetector`: Complete 5-stage Canny Edge Detection with Non-Maximum Suppression and Hysteresis.

### Multithreaded DataLoader (`Datasets/`)
* `ImageFolderDataset`: Parallel directory scanner for image classifications.
* `VisionDataLoader`: Bounded Channel prefetching pipeline with double-buffering.
