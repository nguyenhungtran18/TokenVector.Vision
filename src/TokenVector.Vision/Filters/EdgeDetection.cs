using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Filters;

/// <summary>
/// Spatial gradient and edge detection operators: Sobel X/Y, Sobel Magnitude, Laplacian, and Scharr.
/// </summary>
public static unsafe class EdgeDetection
{
    /// <summary>
    /// Computes Sobel Edge Magnitude: sqrt(Gx^2 + Gy^2).
    /// </summary>
    public static ImageBuffer Sobel(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var target = output ?? new ImageBuffer(input.Info);

        int w = input.Width;
        int h = input.Height;
        int channels = input.Channels;

        byte* srcBase = input.BytePointer;
        byte* dstBase = target.BytePointer;

        for (int y = 0; y < h; y++)
        {
            int y0 = Math.Clamp(y - 1, 0, h - 1);
            int y1 = Math.Clamp(y + 1, 0, h - 1);

            byte* r0 = srcBase + (y0 * input.Stride);
            byte* r1 = srcBase + (y * input.Stride);
            byte* r2 = srcBase + (y1 * input.Stride);
            byte* dstRow = dstBase + (y * target.Stride);

            for (int x = 0; x < w; x++)
            {
                int x0 = Math.Clamp(x - 1, 0, w - 1) * channels;
                int x1 = Math.Clamp(x + 1, 0, w - 1) * channels;
                int xc = x * channels;

                for (int c = 0; c < channels; c++)
                {
                    // Sobel X kernel: [-1 0 1; -2 0 2; -1 0 1]
                    float gx = (-1.0f * r0[x0 + c]) + (1.0f * r0[x1 + c]) +
                               (-2.0f * r1[x0 + c]) + (2.0f * r1[x1 + c]) +
                               (-1.0f * r2[x0 + c]) + (1.0f * r2[x1 + c]);

                    // Sobel Y kernel: [-1 -2 -1; 0 0 0; 1 2 1]
                    float gy = (-1.0f * r0[x0 + c]) + (-2.0f * r0[xc + c]) + (-1.0f * r0[x1 + c]) +
                               ( 1.0f * r2[x0 + c]) + ( 2.0f * r2[xc + c]) + ( 1.0f * r2[x1 + c]);

                    float mag = MathF.Sqrt(gx * gx + gy * gy);
                    dstRow[xc + c] = (byte)Math.Clamp((int)(mag + 0.5f), 0, 255);
                }
            }
        }

        return target;
    }

    /// <summary>
    /// Computes discrete 2D Laplacian second derivative filter.
    /// Kernel: [0, 1, 0; 1, -4, 1; 0, 1, 0]
    /// </summary>
    public static ImageBuffer Laplacian(ImageBuffer input, ImageBuffer? output = null)
    {
        var filter = new Convolution2D(new float[,]
        {
            { 0,  1, 0 },
            { 1, -4, 1 },
            { 0,  1, 0 }
        });
        return filter.Apply(input, output);
    }
}
