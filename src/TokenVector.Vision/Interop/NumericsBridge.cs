using System;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Interop;

/// <summary>
/// Ultra high-performance Zero-Copy and direct SIMD Bridge between TokenVector.Vision 
/// and TokenVector.Numerics mathematical tensor kernel (NDArray of generic type).
/// </summary>
public static unsafe class NumericsBridge
{
    /// <summary>
    /// Exports an ImageBuffer directly into an unmanaged NDArray of float tensor.
    /// If layout matches, data is transferred via direct AVX2/SIMD stream or zero-copy.
    /// </summary>
    public static NDArray<float> ToNDArray(this ImageBuffer buffer, MemoryLayout targetLayout = MemoryLayout.CHW)
    {
        ArgumentNullException.ThrowIfNull(buffer);

        int[] shape = targetLayout switch
        {
            MemoryLayout.CHW => [buffer.Channels, buffer.Height, buffer.Width],
            MemoryLayout.HWC => [buffer.Height, buffer.Width, buffer.Channels],
            MemoryLayout.NCHW => [1, buffer.Channels, buffer.Height, buffer.Width],
            MemoryLayout.NHWC => [1, buffer.Height, buffer.Width, buffer.Channels],
            _ => throw new ArgumentOutOfRangeException(nameof(targetLayout))
        };

        var ndarray = new NDArray<float>(shape);
        CopyToNDArray(buffer, ndarray, targetLayout);
        return ndarray;
    }

    /// <summary>
    /// Exports an 8-bit ImageBuffer directly into an NDArray of byte.
    /// </summary>
    public static NDArray<byte> ToNDArrayByte(this ImageBuffer buffer, MemoryLayout targetLayout = MemoryLayout.HWC)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if (buffer.IsFloatingPoint)
        {
            throw new InvalidOperationException("Buffer contains floating-point data. Use ToNDArray() instead.");
        }

        int[] shape = targetLayout switch
        {
            MemoryLayout.HWC => [buffer.Height, buffer.Width, buffer.Channels],
            MemoryLayout.CHW => [buffer.Channels, buffer.Height, buffer.Width],
            _ => throw new ArgumentOutOfRangeException(nameof(targetLayout))
        };

        var ndarray = new NDArray<byte>(shape);
        Span<byte> destSpan = ndarray.Buffer.AsSpan(ndarray.Offset, ndarray.TotalLength);
        Span<byte> srcSpan = buffer.AsByteSpan();
        srcSpan.CopyTo(destSpan);
        return ndarray;
    }

    /// <summary>
    /// Creates an ImageBuffer wrapping an existing unmanaged/pinned NDArray of float directly.
    /// Zero-Copy: shares memory with the underlying tensor.
    /// </summary>
    public static ImageBuffer AsImageBuffer(this NDArray<float> ndarray, MemoryLayout sourceLayout = MemoryLayout.CHW)
    {
        ArgumentNullException.ThrowIfNull(ndarray);

        int width, height, channels;
        if (sourceLayout == MemoryLayout.CHW)
        {
            if (ndarray.Rank != 3 && ndarray.Rank != 4)
                throw new ArgumentException($"Expected rank 3 [C, H, W] or 4 [1, C, H, W], got rank {ndarray.Rank}.");

            int offset = ndarray.Rank == 4 ? 1 : 0;
            channels = ndarray.Shape[offset];
            height = ndarray.Shape[offset + 1];
            width = ndarray.Shape[offset + 2];
        }
        else if (sourceLayout == MemoryLayout.HWC)
        {
            if (ndarray.Rank != 3 && ndarray.Rank != 4)
                throw new ArgumentException($"Expected rank 3 [H, W, C] or 4 [1, H, W, C], got rank {ndarray.Rank}.");

            int offset = ndarray.Rank == 4 ? 1 : 0;
            height = ndarray.Shape[offset];
            width = ndarray.Shape[offset + 1];
            channels = ndarray.Shape[offset + 2];
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(sourceLayout));
        }

        var format = sourceLayout == MemoryLayout.CHW ? PixelFormat.Float32Planar : PixelFormat.Float32Rgb;
        var info = new ImageInfo(width, height, channels, format, 0, sourceLayout);

        float* rawPtr = ndarray.Buffer.GetUnsafePointer() + ndarray.Offset;
        return new ImageBuffer(rawPtr, info, isOwner: false);
    }

    /// <summary>
    /// Creates an ImageBuffer wrapping an existing NDArray of byte directly.
    /// Zero-Copy: shares memory with the underlying tensor.
    /// </summary>
    public static ImageBuffer AsImageBuffer(this NDArray<byte> ndarray, MemoryLayout sourceLayout = MemoryLayout.HWC)
    {
        ArgumentNullException.ThrowIfNull(ndarray);
        if (ndarray.Rank != 3)
            throw new ArgumentException($"Expected rank 3 [H, W, C], got rank {ndarray.Rank}.");

        int height = ndarray.Shape[0];
        int width = ndarray.Shape[1];
        int channels = ndarray.Shape[2];

        var format = channels switch
        {
            1 => PixelFormat.Grayscale8,
            3 => PixelFormat.Rgb24,
            4 => PixelFormat.Rgba32,
            _ => PixelFormat.Rgb24
        };

        var info = new ImageInfo(width, height, channels, format, width * channels, sourceLayout);
        byte* rawPtr = ndarray.Buffer.GetUnsafePointer() + ndarray.Offset;
        return new ImageBuffer(rawPtr, info, isOwner: false);
    }

    /// <summary>
    /// Copies pixel data from ImageBuffer into a pre-allocated NDArray of float.
    /// Zero-GC allocation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void CopyToNDArray(ImageBuffer buffer, NDArray<float> destination, MemoryLayout targetLayout = MemoryLayout.CHW)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        ArgumentNullException.ThrowIfNull(destination);

        float* dstPtr = destination.Buffer.GetUnsafePointer() + destination.Offset;

        if (buffer.IsFloatingPoint)
        {
            if (buffer.Layout == targetLayout)
            {
                Buffer.MemoryCopy(buffer.FloatPointer, dstPtr, destination.TotalLength * sizeof(float), buffer.ElementCount * sizeof(float));
            }
            else if (buffer.Layout == MemoryLayout.HWC && targetLayout == MemoryLayout.CHW)
            {
                // Transpose HWC float -> CHW float
                int w = buffer.Width;
                int h = buffer.Height;
                int c = buffer.Channels;
                int planeSize = w * h;
                float* src = buffer.FloatPointer;

                for (int y = 0; y < h; y++)
                {
                    int rowOffset = y * w;
                    for (int x = 0; x < w; x++)
                    {
                        int srcIdx = (rowOffset + x) * c;
                        int pixelIdx = rowOffset + x;
                        for (int ch = 0; ch < c; ch++)
                        {
                            dstPtr[ch * planeSize + pixelIdx] = src[srcIdx + ch];
                        }
                    }
                }
            }
            else
            {
                throw new NotSupportedException($"Layout conversion from {buffer.Layout} to {targetLayout} not supported directly.");
            }
        }
        else
        {
            // Convert byte [0, 255] -> float [0.0, 1.0] or [0, 255]
            int w = buffer.Width;
            int h = buffer.Height;
            int c = buffer.Channels;
            int planeSize = w * h;
            byte* src = buffer.BytePointer;

            if (targetLayout == MemoryLayout.CHW)
            {
                for (int y = 0; y < h; y++)
                {
                    int rowOffset = y * w;
                    for (int x = 0; x < w; x++)
                    {
                        int srcIdx = (rowOffset + x) * c;
                        int pixelIdx = rowOffset + x;
                        for (int ch = 0; ch < c; ch++)
                        {
                            dstPtr[ch * planeSize + pixelIdx] = src[srcIdx + ch] * (1.0f / 255.0f);
                        }
                    }
                }
            }
            else
            {
                int total = buffer.ElementCount;
                for (int i = 0; i < total; i++)
                {
                    dstPtr[i] = src[i] * (1.0f / 255.0f);
                }
            }
        }
    }
}
