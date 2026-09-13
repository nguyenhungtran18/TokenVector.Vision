# TokenVector Language Vision Syntax Specification

This specification documents the high-level DSL and language syntax bindings for TokenVector Vision operators within the `.tkv` language runtime.

---

## 1. Vision Module Import

```tkv
import vision as tv_vis;
import numerics as tv_num;
```

---

## 2. Image Loading and Transforms

```tkv
# Load image directly into unmanaged memory buffer
let img = tv_vis.load_image("dataset/cat.jpg");

# Fused 1-Pass preprocessing pipeline
let mean = [0.485, 0.456, 0.406];
let std  = [0.229, 0.224, 0.225];

let tensor_chw = img.fused_resize_normalize_chw(
    width: 224, 
    height: 224, 
    mean: mean, 
    std: std
);

# Zero-copy bridge to numerical tensor
let input_tensor: tv_num.Tensor<f32> = tensor_chw.to_tensor();
```

---

## 3. Spatial Filtering and Edge Detection

```tkv
let blurred = img.gaussian_blur(sigma: 1.5);
let edges   = img.canny_edge(low: 50.0, high: 150.0);
```

---

## 4. Vision Data Loader

```tkv
let ds = tv_vis.ImageFolder("data/imagenet");
let loader = tv_vis.DataLoader(ds, batch_size: 64, shuffle: true);

for batch in loader {
    let (images, labels) = batch;
    # Forward pass on tensor
    let pred = model.forward(images);
}
```
