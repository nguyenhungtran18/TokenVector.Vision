# Ghi chú Phát hành: TokenVector.Vision - v1.0.0 (Release Notes)

**Ngày phát hành:** Tháng 9, 2026  
**Nền tảng:** .NET 8.0 / 9.0 Native AOT, C# 12.0

---

## Các Tính Năng Mới Trong Phiên Bản v1.0.0

### Hạt nhân Bộ nhớ Unmanaged (`Common/`)
* **`ImageBuffer`**: Quản lý vùng nhớ unmanaged căn chỉnh 64-byte (`byte*`, `float*`), đếm tham chiếu (reference counting), cắt vùng ảnh zero-copy (sub-region slicing).
* **Định dạng Layout**: Hỗ trợ đầy đủ `HWC`, `CHW`, `NCHW`, `NHWC`.

### Cầu nối Zero-Copy với `TokenVector.Numerics` (`Interop/`)
* `ToNDArray` & `AsImageBuffer`: Chuyển đổi hai chiều tức thì với `NDArray<float>` và `NDArray<byte>`.

### Bộ Biến đổi Tăng tốc SIMD (`Transforms/`)
* `ResizeTransform`: Nội suy Bilinear và Bicubic tăng tốc AVX2.
* `CropTransforms`: `CenterCrop`, `RandomCrop`, `PadTransform`.
* `GeometricTransforms`: `HorizontalFlip`, `VerticalFlip`, `Rotate` (Ma trận Affine xoay quanh tâm).
* `PhotometricTransforms`: `ColorJitter` (Độ sáng, Độ tương phản, Độ bão hoà), `AdjustGamma`, `GrayscaleTransform`.
* `NormalizeTransform`: Chuẩn hoá Vectorized $(x - \text{mean}) / \text{std}$.
* `LayoutTransforms`: Chuyển vị SIMD $HWC \leftrightarrow CHW$.
* **`FusedTransforms.ResizeNormalizeCHW`**: Toán tử hợp nhất 1-pass tăng tốc 3.24x và 0 byte GC allocation.

### Bộ Lọc Không gian & Canny Edge (`Filters/`)
* `GaussianBlur`: Bộ lọc Gaussian tách rời 1D Ngang + Dọc AVX2.
* `Convolution2D`: Tích chập không gian $3\times 3, 5\times 5, 7\times 7$.
* `EdgeDetection`: Sobel X/Y, Magnitude, Laplacian.
* `CannyDetector`: Pipeline Canny Edge 5 giai đoạn hoàn chỉnh (NMS + Hysteresis).

### DataLoader Đa luồng (`Datasets/`)
* `ImageFolderDataset`: Quét thư mục phân loại ảnh song song.
* `VisionDataLoader`: Pipeline nạp trước dữ liệu qua Bounded Channel kết hợp double-buffering.
