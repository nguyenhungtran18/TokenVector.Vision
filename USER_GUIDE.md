# TokenVector.Vision User Guide

Welcome to the comprehensive user guide for **TokenVector.Vision**. This guide explains how to construct high-performance computer vision pipelines, load datasets, leverage fused operators, and bridge image buffers into mathematical tensor kernels.

---

## 1. ImageBuffer Management and Slicing

`ImageBuffer` is the foundational unmanaged data structure in TokenVector.Vision.

### Creating and Disposing Buffers
```csharp
using TokenVector.Vision.Common;

// Create an RGB24 buffer (HWC layout)
using var rgbImage = ImageBuffer.CreateRgb(width: 640, height: 480);

// Create a Planar Float32 buffer (CHW layout for CNNs)
using var tensorImage = ImageBuffer.CreateFloat32CHW(width: 224, height: 224, channels: 3);
```

### Zero-Copy Sub-region Slicing
```csharp
// Slices a 200x200 patch from (x=100, y=50) sharing parent unmanaged memory
using var cropView = rgbImage.Slice(x: 100, y: 50, width: 200, height: 200);
```

---

## 2. Zero-Copy Tensor Interop with TokenVector.Numerics

Convert seamlessly between `ImageBuffer` and `TokenVector.Numerics.Core.NDArray<T>`.

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Vision.Interop;

// Export ImageBuffer directly into NDArray<float> [3, 224, 224]
using NDArray<float> tensor = tensorImage.ToNDArray(MemoryLayout.CHW);

// Wrap an existing tensor into ImageBuffer without memory copying
using ImageBuffer wrappedBuffer = tensor.AsImageBuffer(MemoryLayout.CHW);
```

---

## 3. Transformations & Pipelines

### Fluent Pipeline API
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

### Fused 1-Pass Operator (Maximum Performance)
```csharp
ReadOnlySpan<float> mean = [0.485f, 0.456f, 0.406f];
ReadOnlySpan<float> std = [0.229f, 0.224f, 0.225f];

// 1 single loop on CPU registers: Resize -> Normalize -> Transpose CHW
using var preprocessed = FusedTransforms.ResizeNormalizeCHW(rgbImage, 224, 224, mean, std);
```

---

## 4. Multi-Threaded DataLoader

```csharp
using TokenVector.Vision.Datasets;

var dataset = new ImageFolderDataset("path/to/imagenet_folder");
await using var loader = new VisionDataLoader(dataset, batchSize: 64, shuffle: true, prefetchCount: 4);

await foreach (var (batchImages, batchLabels) in loader.GetBatchesAsync())
{
    // batchImages has shape [64, 3, 224, 224]
    // batchLabels has shape [64]
    
    // Pass directly to Neural Network model
    // ...
    
    batchImages.Dispose();
    batchLabels.Dispose();
}
```
