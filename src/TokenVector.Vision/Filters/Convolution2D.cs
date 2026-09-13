using System;
using System.Runtime.CompilerServices;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Filters;

/// <summary>
/// High-performance 2D Spatial Convolution Kernel engine.
/// Supports arbitrary kernel dimensions (3x3, 5x5, 7x7) with border replication.
/// </summary>
public sealed unsafe class Convolution2D : IVisionFilter
{
    private readonly float[] _kernel;
    public int KernelWidth { get; }
    public int KernelHeight { get; }

    public Convolution2D(float[,] kernel)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        KernelHeight = kernel.GetLength(0);
        KernelWidth = kernel.GetLength(1);
        if (KernelWidth % 2 == 0 || KernelHeight % 2 == 0)
            throw new ArgumentException("Kernel dimensions must be odd numbers.");

        _kernel = new float[KernelHeight * KernelWidth];
        for (int y = 0; y < KernelHeight; y++)
        {
            for (int x = 0; x < KernelWidth; x++)
            {
                _kernel[y * KernelWidth + x] = kernel[y, x];
            }
        }
    }

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var target = output ?? new ImageBuffer(input.Info);

        int width = input.Width;
        int height = input.Height;
        int channels = input.Channels;

        int radX = KernelWidth / 2;
        int radY = KernelHeight / 2;

        fixed (float* kPtr = _kernel)
        {
            byte* srcBase = input.BytePointer;
            byte* dstBase = target.BytePointer;

            for (int y = 0; y < height; y++)
            {
                byte* dstRow = dstBase + (y * target.Stride);

                for (int x = 0; x < width; x++)
                {
                    int dstOffset = x * channels;

                    for (int c = 0; c < channels; c++)
                    {
                        float sum = 0.0f;

                        for (int ky = -radY; ky <= radY; ky++)
                        {
                            int py = Math.Clamp(y + ky, 0, height - 1);
                            byte* srcRow = srcBase + (py * input.Stride);
                            int kRowOffset = (ky + radY) * KernelWidth;

                            for (int kx = -radX; kx <= radX; kx++)
                            {
                                int px = Math.Clamp(x + kx, 0, width - 1);
                                float weight = kPtr[kRowOffset + (kx + radX)];
                                sum += srcRow[px * channels + c] * weight;
                            }
                        }

                        dstRow[dstOffset + c] = (byte)Math.Clamp((int)(sum + 0.5f), 0, 255);
                    }
                }
            }
        }

        return target;
    }
}
