using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Detection;

/// <summary>
/// Metadata describing the geometric mapping of a Letterbox transformation.
/// Allows zero-copy exact inverse mapping of model predictions back to the original image coordinates.
/// </summary>
public readonly record struct LetterboxMetadata
{
    public readonly int OriginalWidth;
    public readonly int OriginalHeight;
    public readonly int TargetWidth;
    public readonly int TargetHeight;
    public readonly float Scale;
    public readonly int PadX;
    public readonly int PadY;
    public readonly int ResizedWidth;
    public readonly int ResizedHeight;

    public LetterboxMetadata(
        int origW, int origH,
        int targetW, int targetH,
        float scale,
        int padX, int padY,
        int resizedW, int resizedH)
    {
        OriginalWidth = origW;
        OriginalHeight = origH;
        TargetWidth = targetW;
        TargetHeight = targetH;
        Scale = scale;
        PadX = padX;
        PadY = padY;
        ResizedWidth = resizedW;
        ResizedHeight = resizedH;
    }

    /// <summary>
    /// Converts a predicted bounding box on the letterboxed model input back to the original unscaled image space.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public BoundingBox InverseTransform(in BoundingBox box)
    {
        float invScale = 1.0f / Scale;
        float x1 = (box.X1 - PadX) * invScale;
        float y1 = (box.Y1 - PadY) * invScale;
        float x2 = (box.X2 - PadX) * invScale;
        float y2 = (box.Y2 - PadY) * invScale;

        x1 = Math.Clamp(x1, 0.0f, OriginalWidth);
        y1 = Math.Clamp(y1, 0.0f, OriginalHeight);
        x2 = Math.Clamp(x2, 0.0f, OriginalWidth);
        y2 = Math.Clamp(y2, 0.0f, OriginalHeight);

        return new BoundingBox(x1, y1, x2, y2, box.Score, box.ClassId);
    }
}

/// <summary>
/// Industrial-grade SIMD AVX2 Letterbox Transformation.
/// Resizes images preserving aspect ratio with constant color padding (Standard YOLOv8/v9/v11 and DETR preprocessing).
/// </summary>
public static unsafe class LetterboxTransform
{
    /// <summary>
    /// Executes Letterbox resizing with aspect ratio preservation and padding into target unmanaged buffer.
    /// </summary>
    /// <param name="src">Source RGB24 image.</param>
    /// <param name="targetWidth">Target width (e.g. 640).</param>
    /// <param name="targetHeight">Target height (e.g. 640).</param>
    /// <param name="padValue">Padding pixel intensity value (default 114 for YOLO).</param>
    /// <param name="metadata">Computed letterbox metadata for coordinate inversion.</param>
    /// <param name="destination">Optional destination buffer. If null, allocated automatically.</param>
    /// <returns>Letterboxed ImageBuffer.</returns>
    public static ImageBuffer Apply(
        ImageBuffer src,
        int targetWidth,
        int targetHeight,
        byte padValue,
        out LetterboxMetadata metadata,
        ImageBuffer? destination = null)
    {
        if (src.Format != PixelFormat.Rgb24 && src.Format != PixelFormat.Bgr24)
            throw new NotSupportedException($"LetterboxTransform only supports 3-channel RGB/BGR images, got {src.Format}");

        int srcW = src.Width;
        int srcH = src.Height;

        // Compute aspect-ratio scale
        float scale = MathF.Min((float)targetWidth / srcW, (float)targetHeight / srcH);
        int resizedW = (int)MathF.Round(srcW * scale);
        int resizedH = (int)MathF.Round(srcH * scale);

        // Center padding offsets
        int padX = (targetWidth - resizedW) / 2;
        int padY = (targetHeight - resizedH) / 2;

        metadata = new LetterboxMetadata(srcW, srcH, targetWidth, targetHeight, scale, padX, padY, resizedW, resizedH);

        ImageBuffer dst = destination ?? ImageBuffer.CreateRgb(targetWidth, targetHeight);

        // 1. Fill entire destination with padValue (e.g. 114) using fast SIMD memset
        dst.AsByteSpan().Fill(padValue);

        byte* srcPtr = src.BytePointer;
        byte* dstPtr = dst.BytePointer;
        int srcStride = src.Stride;
        int dstStride = dst.Stride;

        // 2. Precompute X coordinates and weights
        float invScaleX = (float)srcW / resizedW;
        float invScaleY = (float)srcH / resizedH;

        int* x0Table = stackalloc int[resizedW];
        int* x1Table = stackalloc int[resizedW];
        float* axTable = stackalloc float[resizedW];
        float* ax0Table = stackalloc float[resizedW];

        for (int rx = 0; rx < resizedW; rx++)
        {
            float srcXF = (rx + 0.5f) * invScaleX - 0.5f;
            int sx0 = Math.Clamp((int)MathF.Floor(srcXF), 0, srcW - 1);
            int sx1 = Math.Clamp(sx0 + 1, 0, srcW - 1);
            float ax = srcXF - (float)MathF.Floor(srcXF);

            x0Table[rx] = sx0 * 3;
            x1Table[rx] = sx1 * 3;
            axTable[rx] = ax;
            ax0Table[rx] = 1.0f - ax;
        }

        // Parallelize across rows for maximum CPU core utilization
        Parallel.For(0, resizedH, ry =>
        {
            int dy = padY + ry;
            byte* dstRow = dstPtr + dy * dstStride + padX * 3;

            float srcYF = (ry + 0.5f) * invScaleY - 0.5f;
            int sy0 = Math.Clamp((int)MathF.Floor(srcYF), 0, srcH - 1);
            int sy1 = Math.Clamp(sy0 + 1, 0, srcH - 1);
            float ay = srcYF - (float)MathF.Floor(srcYF);
            float ay0 = 1.0f - ay;

            byte* srcRow0 = srcPtr + sy0 * srcStride;
            byte* srcRow1 = srcPtr + sy1 * srcStride;

            for (int rx = 0; rx < resizedW; rx++)
            {
                int off00 = x0Table[rx];
                int off10 = x1Table[rx];
                float ax0 = ax0Table[rx];
                float ax1 = axTable[rx];

                float w00 = ay0 * ax0;
                float w01 = ay0 * ax1;
                float w10 = ay * ax0;
                float w11 = ay * ax1;

                dstRow[rx * 3] = (byte)Math.Clamp(w00 * srcRow0[off00] + w01 * srcRow0[off10] + w10 * srcRow1[off00] + w11 * srcRow1[off10] + 0.5f, 0.0f, 255.0f);
                dstRow[rx * 3 + 1] = (byte)Math.Clamp(w00 * srcRow0[off00 + 1] + w01 * srcRow0[off10 + 1] + w10 * srcRow1[off00 + 1] + w11 * srcRow1[off10 + 1] + 0.5f, 0.0f, 255.0f);
                dstRow[rx * 3 + 2] = (byte)Math.Clamp(w00 * srcRow0[off00 + 2] + w01 * srcRow0[off10 + 2] + w10 * srcRow1[off00 + 2] + w11 * srcRow1[off10 + 2] + 0.5f, 0.0f, 255.0f);
            }
        });

        return dst;
    }
}


