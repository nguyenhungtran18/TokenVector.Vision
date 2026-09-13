using System;
using System.Runtime.CompilerServices;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Transforms;

/// <summary>
/// High-speed geometric transforms: Horizontal Flip, Vertical Flip, and Affine Rotation.
/// </summary>
public sealed unsafe class HorizontalFlipTransform : IVisionTransform
{
    public ImageInfo GetOutputInfo(ImageInfo inputInfo) => inputInfo;

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var target = output ?? new ImageBuffer(input.Info);

        int width = input.Width;
        int height = input.Height;
        int channels = input.Channels;

        if (!input.IsFloatingPoint)
        {
            for (int y = 0; y < height; y++)
            {
                byte* srcRow = input.GetRowBytePointer(y);
                byte* dstRow = target.GetRowBytePointer(y);

                if (channels == 3)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int srcX = (width - 1 - x) * 3;
                        int dstX = x * 3;
                        dstRow[dstX] = srcRow[srcX];
                        dstRow[dstX + 1] = srcRow[srcX + 1];
                        dstRow[dstX + 2] = srcRow[srcX + 2];
                    }
                }
                else if (channels == 1)
                {
                    for (int x = 0; x < width; x++)
                    {
                        dstRow[x] = srcRow[width - 1 - x];
                    }
                }
                else
                {
                    for (int x = 0; x < width; x++)
                    {
                        int srcX = (width - 1 - x) * channels;
                        int dstX = x * channels;
                        for (int c = 0; c < channels; c++) dstRow[dstX + c] = srcRow[srcX + c];
                    }
                }
            }
        }
        else
        {
            for (int y = 0; y < height; y++)
            {
                float* srcRow = (float*)input.GetRowBytePointer(y);
                float* dstRow = (float*)target.GetRowBytePointer(y);
                for (int x = 0; x < width; x++)
                {
                    int srcX = (width - 1 - x) * channels;
                    int dstX = x * channels;
                    for (int c = 0; c < channels; c++) dstRow[dstX + c] = srcRow[srcX + c];
                }
            }
        }

        return target;
    }
}

/// <summary>
/// Vertical image flip transform.
/// </summary>
public sealed unsafe class VerticalFlipTransform : IVisionTransform
{
    public ImageInfo GetOutputInfo(ImageInfo inputInfo) => inputInfo;

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var target = output ?? new ImageBuffer(input.Info);

        int height = input.Height;
        int rowBytes = input.Width * input.Channels * (input.IsFloatingPoint ? sizeof(float) : sizeof(byte));

        for (int y = 0; y < height; y++)
        {
            byte* srcRow = input.GetRowBytePointer(height - 1 - y);
            byte* dstRow = target.GetRowBytePointer(y);
            Buffer.MemoryCopy(srcRow, dstRow, target.Stride, rowBytes);
        }

        return target;
    }
}

/// <summary>
/// Continuous affine rotation transform with bilinear resampling.
/// </summary>
public sealed unsafe class RotateTransform : IVisionTransform
{
    public float AngleDegrees { get; }
    public InterpolationMode Mode { get; }

    public RotateTransform(float angleDegrees, InterpolationMode mode = InterpolationMode.Bilinear)
    {
        AngleDegrees = angleDegrees;
        Mode = mode;
    }

    public ImageInfo GetOutputInfo(ImageInfo inputInfo) => inputInfo;

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var target = output ?? new ImageBuffer(input.Info);
        target.Clear();

        float rad = (float)(AngleDegrees * Math.PI / 180.0);
        float cos = (float)Math.Cos(rad);
        float sin = (float)Math.Sin(rad);

        int w = input.Width;
        int h = input.Height;
        int channels = input.Channels;
        float cx = w * 0.5f;
        float cy = h * 0.5f;

        byte* srcBase = input.BytePointer;
        byte* dstBase = target.BytePointer;

        for (int y = 0; y < h; y++)
        {
            float dy = y - cy;
            byte* dstRow = dstBase + (y * target.Stride);

            for (int x = 0; x < w; x++)
            {
                float dx = x - cx;

                // Inverse rotation mapping
                float srcX = cos * dx + sin * dy + cx;
                float srcY = -sin * dx + cos * dy + cy;

                if (srcX >= 0 && srcX < w - 1 && srcY >= 0 && srcY < h - 1)
                {
                    int x0 = (int)srcX;
                    int y0 = (int)srcY;
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    float ax = srcX - x0;
                    float ay = srcY - y0;
                    float w00 = (1.0f - ax) * (1.0f - ay);
                    float w10 = ax * (1.0f - ay);
                    float w01 = (1.0f - ax) * ay;
                    float w11 = ax * ay;

                    byte* p00 = srcBase + (y0 * input.Stride) + (x0 * channels);
                    byte* p10 = srcBase + (y0 * input.Stride) + (x1 * channels);
                    byte* p01 = srcBase + (y1 * input.Stride) + (x0 * channels);
                    byte* p11 = srcBase + (y1 * input.Stride) + (x1 * channels);

                    int dstOffset = x * channels;
                    for (int c = 0; c < channels; c++)
                    {
                        float val = (p00[c] * w00) + (p10[c] * w10) + (p01[c] * w01) + (p11[c] * w11);
                        dstRow[dstOffset + c] = (byte)Math.Clamp((int)(val + 0.5f), 0, 255);
                    }
                }
            }
        }

        return target;
    }
}
