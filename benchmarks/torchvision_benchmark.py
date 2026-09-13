import time
import torch
import torchvision
import torchvision.transforms.v2 as v2
import torchvision.ops as ops
import numpy as np

print("=" * 100)
print(" TORCHVISION vs TOKENVECTOR.VISION - HEAD-TO-HEAD BENCHMARK SUITE")
print(f" PyTorch: {torch.__version__} | TorchVision: {torchvision.__version__} | CPU Threads: {torch.get_num_threads()}")
print("=" * 100)

# --------------------------------------------------------------------------------------------------
# 1. 2D Standard Preprocessing (640x480 -> 224x224 CHW Float32 Normalized)
# --------------------------------------------------------------------------------------------------
print("\n[BENCHMARK 1] Standard 2D Preprocessing (640x480 RGB Byte -> 224x224 CHW Float32 ImageNet)")
print("-" * 100)

raw_img_np = np.full((480, 640, 3), 128, dtype=np.uint8)
img_tensor = torch.from_numpy(raw_img_np).permute(2, 0, 1) # HWC -> CHW

transform_v2 = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR, antialias=False),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Warmup
for _ in range(50):
    _ = transform_v2(img_tensor)

iterations = 500
t0 = time.perf_counter()
for _ in range(iterations):
    _ = transform_v2(img_tensor)
t1 = time.perf_counter()

tv_latency_ms = ((t1 - t0) / iterations) * 1000.0
tv_fps = iterations / (t1 - t0)
print(f"  [TorchVision v2.Compose]: Latency = {tv_latency_ms:.3f} ms | Throughput = {tv_fps:.1f} FPS")

# --------------------------------------------------------------------------------------------------
# 2. 4K UHD Frame Processing (3840x2160 -> 224x224 CHW Tensor)
# --------------------------------------------------------------------------------------------------
print("\n[BENCHMARK 2] 4K UHD Mega-Resolution Preprocessing (3840x2160 -> 224x224 CHW Tensor)")
print("-" * 100)

raw_4k = torch.full((3, 2160, 3840), 128, dtype=torch.uint8)

# Warmup
for _ in range(5):
    _ = transform_v2(raw_4k)

iter_4k = 50
t0 = time.perf_counter()
for _ in range(iter_4k):
    _ = transform_v2(raw_4k)
t1 = time.perf_counter()

tv_4k_ms = ((t1 - t0) / iter_4k) * 1000.0
tv_4k_fps = iter_4k / (t1 - t0)
print(f"  [TorchVision 4K UHD]: Latency = {tv_4k_ms:.3f} ms | Throughput = {tv_4k_fps:.1f} FPS")

# --------------------------------------------------------------------------------------------------
# 3. Fast Non-Maximum Suppression (NMS on 1,000 Candidate Boxes)
# --------------------------------------------------------------------------------------------------
print("\n[BENCHMARK 3] Non-Maximum Suppression (NMS on 1,000 Candidate Bounding Boxes)")
print("-" * 100)

boxes = torch.rand((1000, 4)) * 500.0
boxes[:, 2:] += boxes[:, :2] + 10.0 # ensure x2 > x1, y2 > y1
scores = torch.rand((1000,))
iou_threshold = 0.5

# Warmup
for _ in range(50):
    _ = ops.nms(boxes, scores, iou_threshold)

iter_nms = 1000
t0 = time.perf_counter()
for _ in range(iter_nms):
    _ = ops.nms(boxes, scores, iou_threshold)
t1 = time.perf_counter()

tv_nms_ms = ((t1 - t0) / iter_nms) * 1000.0
tv_nms_ops = iter_nms / (t1 - t0)
print(f"  [TorchVision ops.nms (C++ Extension)]: Latency = {tv_nms_ms:.4f} ms | Throughput = {tv_nms_ops:.1f} NMS ops/sec")

# --------------------------------------------------------------------------------------------------
# 4. 3D Volumetric Trilinear Resampling (128x256x256 -> 64x128x128)
# --------------------------------------------------------------------------------------------------
print("\n[BENCHMARK 4] 3D Volumetric Trilinear Resampling (128x256x256 -> 64x128x128)")
print("-" * 100)

vol_3d = torch.full((1, 1, 128, 256, 256), 1.0, dtype=torch.float32)

# Warmup
for _ in range(5):
    _ = torch.nn.functional.interpolate(vol_3d, size=(64, 128, 128), mode='trilinear', align_corners=False)

iter_3d = 30
t0 = time.perf_counter()
for _ in range(iter_3d):
    _ = torch.nn.functional.interpolate(vol_3d, size=(64, 128, 128), mode='trilinear', align_corners=False)
t1 = time.perf_counter()

tv_3d_ms = ((t1 - t0) / iter_3d) * 1000.0
tv_3d_rate = (64 * 128 * 128 * iter_3d) / ((t1 - t0) * 1_000_000.0)
print(f"  [TorchVision / PyTorch F.interpolate 3D]: Latency = {tv_3d_ms:.2f} ms | Throughput = {tv_3d_rate:.2f} MegaVoxels/sec")

# --------------------------------------------------------------------------------------------------
# 5. Batch 2D Image Preprocessing (Batch N = 128 of 640x480)
# --------------------------------------------------------------------------------------------------
print("\n[BENCHMARK 5] Batch 2D Image Preprocessing (Batch N = 128 of 640x480)")
print("-" * 100)

batch_tensor = torch.full((128, 3, 480, 640), 128, dtype=torch.uint8)

# Warmup
for _ in range(3):
    _ = transform_v2(batch_tensor)

iter_batch = 10
t0 = time.perf_counter()
for _ in range(iter_batch):
    _ = transform_v2(batch_tensor)
t1 = time.perf_counter()

tv_batch_ms = ((t1 - t0) / iter_batch) * 1000.0
tv_batch_fps = (128 * iter_batch) / (t1 - t0)
print(f"  [TorchVision Batch v2 (N=128)]: Latency = {tv_batch_ms:.2f} ms | Throughput = {tv_batch_fps:.1f} FPS")

print("\n" + "=" * 100)
print(" TORCHVISION BENCHMARK COMPLETED SUCCESSFULLY!")
print("=" * 100)
