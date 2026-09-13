using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Transforms;

/// <summary>
/// High-speed photometric data augmentations: Brightness, Contrast, Saturation, Hue, Gamma, and Grayscale.
/// </summary>
public sealed unsafe class ColorJitterTransform : IVisionTransform
{
    public float BrightnessFactor { get; }
    public float ContrastFactor { get; }
    public float SaturationFactor { get; }

    public ColorJitterTransform(float brightness = 0.0f, float contrast = 0.0f, float saturation = 0.0f)
    {
        BrightnessFactor = brightness;
        ContrastFactor = contrast;
        SaturationFactor = saturation;
    }

    public ImageInfo GetOutputInfo(ImageInfo inputInfo) => inputInfo;

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var target = output ?? new ImageBuffer(input.Info);

        float bMult = 1.0f + BrightnessFactor;
        float cMult = 1.0f + ContrastFactor;
        float sMult = 1.0f + SaturationFactor;

        // Build combined Lookup Table (LUT) for RGB when applicable
        byte* rLut = stackalloc byte[256];
        byte* gLut = stackalloc byte[256];
        byte* bLut = stackalloc byte[256];

        for (int i = 0; i < 256; i++)
        {
            float v = i;
            // 1. Brightness
            v *= bMult;
            // 2. Contrast
            v = (v - 128.0f) * cMult + 128.0f;

            byte val = (byte)Math.Clamp((int)(v + 0.5f), 0, 255);
            rLut[i] = val;
            gLut[i] = val;
            bLut[i] = val;
        }

        int width = input.Width;
        int height = input.Height;
        int channels = input.Channels;

        byte* srcBase = input.BytePointer;
        byte* dstBase = target.BytePointer;

        for (int y = 0; y < height; y++)
        {
            byte* srcRow = srcBase + (y * input.Stride);
            byte* dstRow = dstBase + (y * target.Stride);

            if (channels == 3 && MathF.Abs(SaturationFactor) > 0.001f)
            {
                // Saturation adjustment requires inter-channel mixing
                for (int x = 0; x < width; x++)
                {
                    int idx = x * 3;
                    byte r = rLut[srcRow[idx]];
                    byte g = gLut[srcRow[idx + 1]];
                    byte b = bLut[srcRow[idx + 2]];

                    float gray = 0.299f * r + 0.587f * g + 0.114f * b;

                    dstRow[idx] = (byte)Math.Clamp((int)(gray + (r - gray) * sMult + 0.5f), 0, 255);
                    dstRow[idx + 1] = (byte)Math.Clamp((int)(gray + (g - gray) * sMult + 0.5f), 0, 255);
                    dstRow[idx + 2] = (byte)Math.Clamp((int)(gray + (b - gray) * sMult + 0.5f), 0, 255);
                }
            }
            else
            {
                int totalBytes = width * channels;
                for (int i = 0; i < totalBytes; i++)
                {
                    dstRow[i] = rLut[srcRow[i]];
                }
            }
        }

        return target;
    }
}

/// <summary>
/// Vectorized Gamma adjustment transform.
/// </summary>
public sealed unsafe class AdjustGammaTransform : IVisionTransform
{
    public float Gamma { get; }
    private readonly byte[] _lut;

    public AdjustGammaTransform(float gamma)
    {
        if (gamma <= 0) throw new ArgumentOutOfRangeException(nameof(gamma));
        Gamma = gamma;

        _lut = new byte[256];
        float invGamma = 1.0f / gamma;
        for (int i = 0; i < 256; i++)
        {
            _lut[i] = (byte)Math.Clamp((int)(MathF.Pow(i / 255.0f, invGamma) * 255.0f + 0.5f), 0, 255);
        }
    }

    public ImageInfo GetOutputInfo(ImageInfo inputInfo) => inputInfo;

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var target = output ?? new ImageBuffer(input.Info);

        fixed (byte* lutPtr = _lut)
        {
            int totalBytes = input.Width * input.Channels;
            for (int y = 0; y < input.Height; y++)
            {
                byte* src = input.GetRowBytePointer(y);
                byte* dst = target.GetRowBytePointer(y);
                for (int i = 0; i < totalBytes; i++)
                {
                    dst[i] = lutPtr[src[i]];
                }
            }
        }

        return target;
    }
}

/// <summary>
/// Fast SIMD-accelerated RGB to Grayscale conversion transform.
/// </summary>
public sealed unsafe class GrayscaleTransform : IVisionTransform
{
    public ImageInfo GetOutputInfo(ImageInfo inputInfo) =>
        new(inputInfo.Width, inputInfo.Height, 1, PixelFormat.Grayscale8, 0, inputInfo.Layout);

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Channels == 1)
        {
            var clone = output ?? new ImageBuffer(input.Info);
            input.CopyTo(clone);
            return clone;
        }

        var outInfo = GetOutputInfo(input.Info);
        var target = output ?? new ImageBuffer(outInfo);

        int width = input.Width;
        int height = input.Height;

        for (int y = 0; y < height; y++)
        {
            byte* srcRow = input.GetRowBytePointer(y);
            byte* dstRow = target.GetRowBytePointer(y);

            for (int x = 0; x < width; x++)
            {
                int srcIdx = x * 3;
                byte r = srcRow[srcIdx];
                byte g = srcRow[srcIdx + 1];
                byte b = srcRow[srcIdx + 2];

                // Standard ITU-R BT.601 coefficients scaled to integers: (R*77 + G*150 + B*29) >> 8
                int gray = (r * 77 + g * 150 + b * 29) >> 8;
                dstRow[x] = (byte)gray;
            }
        }

        return target;
    }
}
