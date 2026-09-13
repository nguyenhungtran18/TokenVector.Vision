# Hướng dẫn Sử dụng TokenVector.Vision (User Guide)

Chào mừng bạn đến với tài liệu hướng dẫn sử dụng thư viện **TokenVector.Vision**. Tài liệu này cung cấp hướng dẫn chi tiết về cách xây dựng các pipeline xử lý ảnh hiệu năng cao, nạp tập dữ liệu huấn luyện, áp dụng toán tử hợp nhất (Fused Operators) và liên kết Zero-Copy với hạt nhân tensor.

---

## 1. Quản lý Bộ nhớ Native ImageBuffer

`ImageBuffer` là cấu trúc dữ liệu unmanaged cơ sở của TokenVector.Vision.

### Khởi tạo và Giải phóng Vùng nhớ
```csharp
using TokenVector.Vision.Common;

// Khởi tạo buffer RGB24 (Layout HWC)
using var rgbImage = ImageBuffer.CreateRgb(width: 640, height: 480);

// Khởi tạo buffer Float32 Planar (Layout CHW cho mạng CNN/ViT)
using var tensorImage = ImageBuffer.CreateFloat32CHW(width: 224, height: 224, channels: 3);
```

### Cắt vùng ảnh Zero-Copy (Sub-region Slicing)
```csharp
// Trích xuất vùng ảnh 200x200 từ toạ độ (x=100, y=50) dùng chung vùng nhớ gốc
using var cropView = rgbImage.Slice(x: 100, y: 50, width: 200, height: 200);
```

---

## 2. Cầu nối Zero-Copy với `TokenVector.Numerics`

Chuyển đổi hai chiều mượt mà giữa `ImageBuffer` và `NDArray<T>`.

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Vision.Interop;

// Xuất trực tiếp ImageBuffer sang tensor NDArray<float> [3, 224, 224]
using NDArray<float> tensor = tensorImage.ToNDArray(MemoryLayout.CHW);

// Bọc một tensor có sẵn thành ImageBuffer mà không copy dữ liệu
using ImageBuffer wrappedBuffer = tensor.AsImageBuffer(MemoryLayout.CHW);
```

---

## 3. Pipeline Biến đổi & Toán tử Fused 1-Pass

### Xây dựng Pipeline Biến đổi
```csharp
using TokenVector.Vision.Transforms;

var pipeline = new ComposePipeline()
    .Resize(256, 256, InterpolationMode.Bilinear)
    .CenterCrop(224, 224)
    .ColorJitter(brightness: 0.1f, contrast: 0.2f)
    .NormalizeImageNet()
    .ToCHW();

using var result = pipeline.Execute(rgbImage);
```

### Toán tử Fused 1-Pass Tối ưu Tuyệt đối
```csharp
ReadOnlySpan<float> mean = [0.485f, 0.456f, 0.406f];
ReadOnlySpan<float> std = [0.229f, 0.224f, 0.225f];

// 1 vòng lặp duy nhất trên thanh ghi CPU: Resize -> Normalize -> Transpose CHW
using var preprocessed = FusedTransforms.ResizeNormalizeCHW(rgbImage, 224, 224, mean, std);
```

---

## 4. Nạp Dữ liệu Đa luồng (DataLoader)

```csharp
using TokenVector.Vision.Datasets;

var dataset = new ImageFolderDataset("path/to/dataset");
await using var loader = new VisionDataLoader(dataset, batchSize: 64, shuffle: true, prefetchCount: 4);

await foreach (var (batchImages, batchLabels) in loader.GetBatchesAsync())
{
    // batchImages có kích thước tensor: [64, 3, 224, 224]
    // batchLabels có kích thước: [64]
    
    // Đưa trực tiếp vào model suy luận hoặc huấn luyện
    // ...
    
    batchImages.Dispose();
    batchLabels.Dispose();
}
```
