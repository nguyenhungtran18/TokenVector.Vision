# Đặc tả Cú pháp Ngôn ngữ TokenVector: Phân hệ Thị giác (Vision)

Tài liệu này quy định cú pháp DSL và các ràng buộc ngôn ngữ cấp cao cho các toán tử Vision trong runtime ngôn ngữ lập trình TokenVector (`.tkv`).

---

## 1. Khai báo Import Phân hệ Vision

```tkv
import vision as tv_vis;
import numerics as tv_num;
```

---

## 2. Tải Ảnh và Biến đổi Tiền xử lý

```tkv
# Nạp ảnh trực tiếp vào bộ nhớ unmanaged
let img = tv_vis.load_image("dataset/cat.jpg");

# Pipeline hợp nhất 1-Pass siêu tốc
let mean = [0.485, 0.456, 0.406];
let std  = [0.229, 0.224, 0.225];

let tensor_chw = img.fused_resize_normalize_chw(
    width: 224, 
    height: 224, 
    mean: mean, 
    std: std
);

# Cầu nối Zero-Copy sang Tensor số học
let input_tensor: tv_num.Tensor<f32> = tensor_chw.to_tensor();
```

---

## 3. Lọc Không gian & Nhận diện Cạnh Canny

```tkv
let blurred = img.gaussian_blur(sigma: 1.5);
let edges   = img.canny_edge(low: 50.0, high: 150.0);
```

---

## 4. DataLoader Nạp Dữ liệu Huấn luyện

```tkv
let ds = tv_vis.ImageFolder("data/imagenet");
let loader = tv_vis.DataLoader(ds, batch_size: 64, shuffle: true);

for batch in loader {
    let (images, labels) = batch;
    # Lan truyền xuôi trên tensor
    let pred = model.forward(images);
}
```
