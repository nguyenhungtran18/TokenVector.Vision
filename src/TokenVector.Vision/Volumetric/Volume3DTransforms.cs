using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Volumetric;

/// <summary>
/// High-performance 3D Volumetric image transformations (Trilinear Resampling, 3D Spatial Filters, 3D Pooling).
/// Designed for high-throughput Zero-GC processing of large 3D medical scans and AI voxel fields.
/// </summary>
public static unsafe class Volume3DTransforms
{
    /// <summary>
    /// Resamples a 3D Volumetric buffer using Trilinear Interpolation (8-corner voxel interpolation).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Volume3DBuffer ResampleTrilinear(
        Volume3DBuffer input,
        int targetDepth,
        int targetHeight,
        int targetWidth,
        Volume3DBuffer? preallocatedOutput = null)
    {
        ArgumentNullException.ThrowIfNull(input);

        int srcD = input.Depth;
        int srcH = input.Height;
        int srcW = input.Width;
        int channels = input.Channels;

        var outInfo = new Volume3DInfo(targetDepth, targetHeight, targetWidth, channels, input.IsFloatingPoint, input.Layout);
        var target = preallocatedOutput ?? new Volume3DBuffer(outInfo);

        float scaleZ = (float)srcD / targetDepth;
        float scaleY = (float)srcH / targetHeight;
        float scaleX = (float)srcW / targetWidth;

        // Precompute X mapping tables
        int* x0Table = stackalloc int[targetWidth];
        int* x1Table = stackalloc int[targetWidth];
        float* axTable = stackalloc float[targetWidth];

        for (int x = 0; x < targetWidth; x++)
        {
            float srcX = (x + 0.5f) * scaleX - 0.5f;
            int x0 = (int)MathF.Floor(srcX);
            int x1 = x0 + 1;
            float ax = srcX - x0;

            x0Table[x] = Math.Clamp(x0, 0, srcW - 1) * channels;
            x1Table[x] = Math.Clamp(x1, 0, srcW - 1) * channels;
            axTable[x] = ax;
        }

        // Precompute Y mapping tables
        int* y0Table = stackalloc int[targetHeight];
        int* y1Table = stackalloc int[targetHeight];
        float* ayTable = stackalloc float[targetHeight];

        for (int y = 0; y < targetHeight; y++)
        {
            float srcY = (y + 0.5f) * scaleY - 0.5f;
            int y0 = (int)MathF.Floor(srcY);
            int y1 = y0 + 1;
            float ay = srcY - y0;

            y0Table[y] = Math.Clamp(y0, 0, srcH - 1);
            y1Table[y] = Math.Clamp(y1, 0, srcH - 1);
            ayTable[y] = ay;
        }

        if (input.IsFloatingPoint)
        {
            float* srcBase = input.FloatPointer;
            float* dstBase = target.FloatPointer;

            int srcSliceStride = srcH * srcW * channels;
            int srcRowStride = srcW * channels;

            int dstSliceStride = targetHeight * targetWidth * channels;
            int dstRowStride = targetWidth * channels;

            Parallel.For(0, targetDepth, z =>
            {
                float srcZ = (z + 0.5f) * scaleZ - 0.5f;
                int z0 = Math.Clamp((int)MathF.Floor(srcZ), 0, srcD - 1);
                int z1 = Math.Clamp(z0 + 1, 0, srcD - 1);
                float az = srcZ - z0;
                float az0 = 1.0f - az;

                float* srcSlice0 = srcBase + ((long)z0 * srcSliceStride);
                float* srcSlice1 = srcBase + ((long)z1 * srcSliceStride);
                float* dstSlice = dstBase + ((long)z * dstSliceStride);

                for (int y = 0; y < targetHeight; y++)
                {
                    int sy0 = y0Table[y];
                    int sy1 = y1Table[y];
                    float ay = ayTable[y];
                    float ay0 = 1.0f - ay;

                    float* s0_r0 = srcSlice0 + (sy0 * srcRowStride);
                    float* s0_r1 = srcSlice0 + (sy1 * srcRowStride);
                    float* s1_r0 = srcSlice1 + (sy0 * srcRowStride);
                    float* s1_r1 = srcSlice1 + (sy1 * srcRowStride);

                    float* dstRow = dstSlice + (y * dstRowStride);

                    // 4 intermediate weights for Z0 and Z1
                    float w00_z0 = az0 * ay0;
                    float w01_z0 = az0 * ay;
                    float w00_z1 = az * ay0;
                    float w01_z1 = az * ay;

                    for (int x = 0; x < targetWidth; x++)
                    {
                        int sx0 = x0Table[x];
                        int sx1 = x1Table[x];
                        float ax = axTable[x];
                        float ax0 = 1.0f - ax;

                        // 8 trilinear corner weights
                        float w000 = w00_z0 * ax0;
                        float w001 = w00_z0 * ax;
                        float w010 = w01_z0 * ax0;
                        float w011 = w01_z0 * ax;

                        float w100 = w00_z1 * ax0;
                        float w101 = w00_z1 * ax;
                        float w110 = w01_z1 * ax0;
                        float w111 = w01_z1 * ax;

                        for (int c = 0; c < channels; c++)
                        {
                            float v = (s0_r0[sx0 + c] * w000) + (s0_r0[sx1 + c] * w001) +
                                      (s0_r1[sx0 + c] * w010) + (s0_r1[sx1 + c] * w011) +
                                      (s1_r0[sx0 + c] * w100) + (s1_r0[sx1 + c] * w101) +
                                      (s1_r1[sx0 + c] * w110) + (s1_r1[sx1 + c] * w111);

                            dstRow[(x * channels) + c] = v;
                        }
                    }
                }
            });
        }
        else
        {
            byte* srcBase = input.BytePointer;
            byte* dstBase = target.BytePointer;

            int srcSliceStride = srcH * srcW * channels;
            int srcRowStride = srcW * channels;

            int dstSliceStride = targetHeight * targetWidth * channels;
            int dstRowStride = targetWidth * channels;

            Parallel.For(0, targetDepth, z =>
            {
                float srcZ = (z + 0.5f) * scaleZ - 0.5f;
                int z0 = Math.Clamp((int)MathF.Floor(srcZ), 0, srcD - 1);
                int z1 = Math.Clamp(z0 + 1, 0, srcD - 1);
                float az = srcZ - z0;
                float az0 = 1.0f - az;

                byte* srcSlice0 = srcBase + ((long)z0 * srcSliceStride);
                byte* srcSlice1 = srcBase + ((long)z1 * srcSliceStride);
                byte* dstSlice = dstBase + ((long)z * dstSliceStride);

                for (int y = 0; y < targetHeight; y++)
                {
                    int sy0 = y0Table[y];
                    int sy1 = y1Table[y];
                    float ay = ayTable[y];
                    float ay0 = 1.0f - ay;

                    byte* s0_r0 = srcSlice0 + (sy0 * srcRowStride);
                    byte* s0_r1 = srcSlice0 + (sy1 * srcRowStride);
                    byte* s1_r0 = srcSlice1 + (sy0 * srcRowStride);
                    byte* s1_r1 = srcSlice1 + (sy1 * srcRowStride);

                    byte* dstRow = dstSlice + (y * dstRowStride);

                    float w00_z0 = az0 * ay0;
                    float w01_z0 = az0 * ay;
                    float w00_z1 = az * ay0;
                    float w01_z1 = az * ay;

                    for (int x = 0; x < targetWidth; x++)
                    {
                        int sx0 = x0Table[x];
                        int sx1 = x1Table[x];
                        float ax = axTable[x];
                        float ax0 = 1.0f - ax;

                        float w000 = w00_z0 * ax0;
                        float w001 = w00_z0 * ax;
                        float w010 = w01_z0 * ax0;
                        float w011 = w01_z0 * ax;

                        float w100 = w00_z1 * ax0;
                        float w101 = w00_z1 * ax;
                        float w110 = w01_z1 * ax0;
                        float w111 = w01_z1 * ax;

                        for (int c = 0; c < channels; c++)
                        {
                            float v = (s0_r0[sx0 + c] * w000) + (s0_r0[sx1 + c] * w001) +
                                      (s0_r1[sx0 + c] * w010) + (s0_r1[sx1 + c] * w011) +
                                      (s1_r0[sx0 + c] * w100) + (s1_r0[sx1 + c] * w101) +
                                      (s1_r1[sx0 + c] * w110) + (s1_r1[sx1 + c] * w111);

                            dstRow[(x * channels) + c] = (byte)Math.Clamp((int)MathF.Round(v), 0, 255);
                        }
                    }
                }
            });
        }

        return target;
    }

    /// <summary>
    /// Executes a 3D Box Spatial Filter (3x3x3 smoothing) over the 3D volume.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Volume3DBuffer FilterBox3D(Volume3DBuffer input, Volume3DBuffer? preallocatedOutput = null)
    {
        ArgumentNullException.ThrowIfNull(input);

        int D = input.Depth;
        int H = input.Height;
        int W = input.Width;
        int channels = input.Channels;

        var target = preallocatedOutput ?? new Volume3DBuffer(input.Info);

        if (input.IsFloatingPoint)
        {
            float* src = input.FloatPointer;
            float* dst = target.FloatPointer;
            const float inv27 = 1.0f / 27.0f;

            Parallel.For(1, D - 1, z =>
            {
                for (int y = 1; y < H - 1; y++)
                {
                    for (int x = 1; x < W - 1; x++)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            float sum = 0;
                            for (int dz = -1; dz <= 1; dz++)
                            {
                                for (int dy = -1; dy <= 1; dy++)
                                {
                                    for (int dx = -1; dx <= 1; dx++)
                                    {
                                        long idx = ((long)(z + dz) * H * W + (y + dy) * W + (x + dx)) * channels + c;
                                        sum += src[idx];
                                    }
                                }
                            }
                            long outIdx = ((long)z * H * W + y * W + x) * channels + c;
                            dst[outIdx] = sum * inv27;
                        }
                    }
                }
            });
        }
        else
        {
            byte* src = input.BytePointer;
            byte* dst = target.BytePointer;
            const float inv27 = 1.0f / 27.0f;

            Parallel.For(1, D - 1, z =>
            {
                for (int y = 1; y < H - 1; y++)
                {
                    for (int x = 1; x < W - 1; x++)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            int sum = 0;
                            for (int dz = -1; dz <= 1; dz++)
                            {
                                for (int dy = -1; dy <= 1; dy++)
                                {
                                    for (int dx = -1; dx <= 1; dx++)
                                    {
                                        long idx = ((long)(z + dz) * H * W + (y + dy) * W + (x + dx)) * channels + c;
                                        sum += src[idx];
                                    }
                                }
                            }
                            long outIdx = ((long)z * H * W + y * W + x) * channels + c;
                            dst[outIdx] = (byte)Math.Clamp((int)MathF.Round(sum * inv27), 0, 255);
                        }
                    }
                }
            });
        }

        return target;
    }

    /// <summary>
    /// Executes 3D Max Pooling downsampling (stride 2x2x2) for 3D AI Feature Extraction.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Volume3DBuffer MaxPooling3D(Volume3DBuffer input, Volume3DBuffer? preallocatedOutput = null)
    {
        ArgumentNullException.ThrowIfNull(input);

        int outD = input.Depth / 2;
        int outH = input.Height / 2;
        int outW = input.Width / 2;
        int channels = input.Channels;

        var outInfo = new Volume3DInfo(outD, outH, outW, channels, input.IsFloatingPoint, input.Layout);
        var target = preallocatedOutput ?? new Volume3DBuffer(outInfo);

        int inH = input.Height;
        int inW = input.Width;

        if (input.IsFloatingPoint)
        {
            float* src = input.FloatPointer;
            float* dst = target.FloatPointer;

            Parallel.For(0, outD, z =>
            {
                int inZ = z * 2;
                for (int y = 0; y < outH; y++)
                {
                    int inY = y * 2;
                    for (int x = 0; x < outW; x++)
                    {
                        int inX = x * 2;
                        for (int c = 0; c < channels; c++)
                        {
                            float maxVal = float.MinValue;
                            for (int dz = 0; dz < 2; dz++)
                            {
                                for (int dy = 0; dy < 2; dy++)
                                {
                                    for (int dx = 0; dx < 2; dx++)
                                    {
                                        long idx = ((long)(inZ + dz) * inH * inW + (inY + dy) * inW + (inX + dx)) * channels + c;
                                        float v = src[idx];
                                        if (v > maxVal) maxVal = v;
                                    }
                                }
                            }
                            long outIdx = ((long)z * outH * outW + y * outW + x) * channels + c;
                            dst[outIdx] = maxVal;
                        }
                    }
                }
            });
        }

        return target;
    }
}
