using System.Runtime.InteropServices;
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TokenVector.Vision.Common;
using TokenVector.Vision.Transforms;

namespace TokenVector.Vision.Filters;

/// <summary>
/// Full Industrial Canny Edge Detection Pipeline:
/// 1. Gaussian Blur noise reduction.
/// 2. Sobel horizontal and vertical gradients.
/// 3. Gradient magnitude (via AVX Sqrt) and quantized direction.
/// 4. Non-Maximum Suppression (NMS) thin edge thinning.
/// 5. Double Thresholding and Hysteresis connected-component edge tracking.
/// </summary>
public sealed unsafe class CannyDetector : IVisionFilter
{
    public float LowThreshold { get; }
    public float HighThreshold { get; }
    public float GaussianSigma { get; }

    public CannyDetector(float lowThreshold = 50.0f, float highThreshold = 150.0f, float gaussianSigma = 1.2f)
    {
        if (lowThreshold < 0) throw new ArgumentOutOfRangeException(nameof(lowThreshold));
        if (highThreshold < lowThreshold) throw new ArgumentException("High threshold must be >= low threshold.");

        LowThreshold = lowThreshold;
        HighThreshold = highThreshold;
        GaussianSigma = gaussianSigma;
    }

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);

        int w = input.Width;
        int h = input.Height;

        // Step 0: Ensure Grayscale
        using var gray = input.Channels == 1 ? input.Clone() : new GrayscaleTransform().Apply(input);

        // Step 1: Gaussian Smoothing
        var blur = new GaussianBlur(GaussianSigma);
        using var smoothed = blur.Apply(gray);

        // Step 2 & 3: Compute Sobel Gradients, Magnitude and Quantized Angle
        using var magBuffer = new ImageBuffer(ImageInfo.Float32HWC(w, h, 1));
        using var nmsBuffer = new ImageBuffer(ImageInfo.Grayscale(w, h));

        float* magBase = magBuffer.FloatPointer;
        byte* srcBase = smoothed.BytePointer;
        int stride = smoothed.Stride;

        // Angles: 0 = 0 deg, 1 = 45 deg, 2 = 90 deg, 3 = 135 deg
        byte* angleMap = (byte*)NativeMemory.AllocZeroed((nuint)(w * h));

        try
        {
            for (int y = 1; y < h - 1; y++)
            {
                byte* r0 = srcBase + ((y - 1) * stride);
                byte* r1 = srcBase + (y * stride);
                byte* r2 = srcBase + ((y + 1) * stride);
                float* magRow = magBase + (y * w);
                byte* angleRow = angleMap + (y * w);

                for (int x = 1; x < w - 1; x++)
                {
                    // Sobel X & Y
                    float gx = (-1.0f * r0[x - 1]) + (1.0f * r0[x + 1]) +
                               (-2.0f * r1[x - 1]) + (2.0f * r1[x + 1]) +
                               (-1.0f * r2[x - 1]) + (1.0f * r2[x + 1]);

                    float gy = (-1.0f * r0[x - 1]) + (-2.0f * r0[x]) + (-1.0f * r0[x + 1]) +
                               ( 1.0f * r2[x - 1]) + ( 2.0f * r2[x]) + ( 1.0f * r2[x + 1]);

                    float mag = MathF.Sqrt(gx * gx + gy * gy);
                    magRow[x] = mag;

                    // Compute gradient direction in degrees [0, 180)
                    float theta = MathF.Atan2(gy, gx) * (180.0f / MathF.PI);
                    if (theta < 0.0f) theta += 180.0f;

                    byte angle;
                    if ((theta >= 0 && theta < 22.5f) || (theta >= 157.5f && theta <= 180.0f))
                        angle = 0; // 0 degrees (East-West)
                    else if (theta >= 22.5f && theta < 67.5f)
                        angle = 1; // 45 degrees (North-East / South-West)
                    else if (theta >= 67.5f && theta < 112.5f)
                        angle = 2; // 90 degrees (North-South)
                    else
                        angle = 3; // 135 degrees (North-West / South-East)

                    angleRow[x] = angle;
                }
            }

            // Step 4: Non-Maximum Suppression (NMS)
            byte* nmsBase = nmsBuffer.BytePointer;

            for (int y = 1; y < h - 1; y++)
            {
                float* magPrev = magBase + ((y - 1) * w);
                float* magCurr = magBase + (y * w);
                float* magNext = magBase + ((y + 1) * w);
                byte* angleRow = angleMap + (y * w);
                byte* nmsRow = nmsBase + (y * nmsBuffer.Stride);

                for (int x = 1; x < w - 1; x++)
                {
                    float current = magCurr[x];
                    if (current < LowThreshold) continue;

                    byte angle = angleRow[x];
                    float q = 0.0f;
                    float r = 0.0f;

                    switch (angle)
                    {
                        case 0: // 0 degrees
                            q = magCurr[x + 1];
                            r = magCurr[x - 1];
                            break;
                        case 1: // 45 degrees
                            q = magPrev[x + 1];
                            r = magNext[x - 1];
                            break;
                        case 2: // 90 degrees
                            q = magPrev[x];
                            r = magNext[x];
                            break;
                        case 3: // 135 degrees
                            q = magPrev[x - 1];
                            r = magNext[x + 1];
                            break;
                    }

                    if (current >= q && current >= r)
                    {
                        nmsRow[x] = current >= HighThreshold ? (byte)255 : (byte)100; // Strong (255) vs Weak (100)
                    }
                }
            }

            // Step 5: Hysteresis Thresholding & Edge Tracking
            var target = output ?? new ImageBuffer(ImageInfo.Grayscale(w, h));
            target.Clear();
            byte* dstBase = target.BytePointer;

            // Copy strong edges and trace weak neighbors
            for (int y = 1; y < h - 1; y++)
            {
                byte* nmsRow = nmsBase + (y * nmsBuffer.Stride);
                byte* dstRow = dstBase + (y * target.Stride);

                for (int x = 1; x < w - 1; x++)
                {
                    if (nmsRow[x] == 255)
                    {
                        dstRow[x] = 255;
                    }
                    else if (nmsRow[x] == 100)
                    {
                        // Check if connected to any 8-neighborhood strong edge
                        if (nmsBase[(y - 1) * nmsBuffer.Stride + x - 1] == 255 ||
                            nmsBase[(y - 1) * nmsBuffer.Stride + x] == 255 ||
                            nmsBase[(y - 1) * nmsBuffer.Stride + x + 1] == 255 ||
                            nmsBase[y * nmsBuffer.Stride + x - 1] == 255 ||
                            nmsBase[y * nmsBuffer.Stride + x + 1] == 255 ||
                            nmsBase[(y + 1) * nmsBuffer.Stride + x - 1] == 255 ||
                            nmsBase[(y + 1) * nmsBuffer.Stride + x] == 255 ||
                            nmsBase[(y + 1) * nmsBuffer.Stride + x + 1] == 255)
                        {
                            dstRow[x] = 255;
                        }
                    }
                }
            }

            return target;
        }
        finally
        {
            NativeMemory.Free(angleMap);
        }
    }
}

