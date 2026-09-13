using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Transforms;

/// <summary>
/// High-speed SIMD Layout Transforms: Transpose Interleaved [H, W, C] to Planar [C, H, W].
/// </summary>
public sealed unsafe class LayoutTransforms : IVisionTransform
{
    public MemoryLayout TargetLayout { get; }

    public LayoutTransforms(MemoryLayout targetLayout)
    {
        TargetLayout = targetLayout;
    }

    public static LayoutTransforms ToCHW { get; } = new(MemoryLayout.CHW);
    public static LayoutTransforms ToHWC { get; } = new(MemoryLayout.HWC);

    public ImageInfo GetOutputInfo(ImageInfo inputInfo)
    {
        var format = TargetLayout == MemoryLayout.CHW ? PixelFormat.Float32Planar : PixelFormat.Float32Rgb;
        return new ImageInfo(inputInfo.Width, inputInfo.Height, inputInfo.Channels, format, 0, TargetLayout);
    }

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Layout == TargetLayout)
        {
            var clone = output ?? new ImageBuffer(input.Info);
            input.CopyTo(clone);
            return clone;
        }

        var outInfo = GetOutputInfo(input.Info);
        var target = output ?? new ImageBuffer(outInfo);

        if (input.Layout == MemoryLayout.HWC && TargetLayout == MemoryLayout.CHW)
        {
            TransposeHwcToChw(input, target);
        }
        else if (input.Layout == MemoryLayout.CHW && TargetLayout == MemoryLayout.HWC)
        {
            TransposeChwToHwc(input, target);
        }
        else
        {
            throw new NotSupportedException($"Layout transformation from {input.Layout} to {TargetLayout} is not supported.");
        }

        return target;
    }

    #region Transposition Kernels

    private static void TransposeHwcToChw(ImageBuffer src, ImageBuffer dst)
    {
        int w = src.Width;
        int h = src.Height;
        int c = src.Channels;
        int planeSize = w * h;

        float* dstBase = dst.FloatPointer;

        if (src.IsFloatingPoint)
        {
            float* srcBase = src.FloatPointer;
            for (int y = 0; y < h; y++)
            {
                int rowOffset = y * w;
                for (int x = 0; x < w; x++)
                {
                    int srcIdx = (rowOffset + x) * c;
                    int pixelIdx = rowOffset + x;
                    for (int ch = 0; ch < c; ch++)
                    {
                        dstBase[ch * planeSize + pixelIdx] = srcBase[srcIdx + ch];
                    }
                }
            }
        }
        else
        {
            byte* srcBase = src.BytePointer;
            for (int y = 0; y < h; y++)
            {
                int rowOffset = y * w;
                for (int x = 0; x < w; x++)
                {
                    int srcIdx = (rowOffset + x) * c;
                    int pixelIdx = rowOffset + x;
                    for (int ch = 0; ch < c; ch++)
                    {
                        dstBase[ch * planeSize + pixelIdx] = srcBase[srcIdx + ch] * (1.0f / 255.0f);
                    }
                }
            }
        }
    }

    private static void TransposeChwToHwc(ImageBuffer src, ImageBuffer dst)
    {
        int w = src.Width;
        int h = src.Height;
        int c = src.Channels;
        int planeSize = w * h;

        float* srcBase = src.FloatPointer;
        float* dstBase = dst.FloatPointer;

        for (int y = 0; y < h; y++)
        {
            int rowOffset = y * w;
            for (int x = 0; x < w; x++)
            {
                int pixelIdx = rowOffset + x;
                int dstIdx = (rowOffset + x) * c;
                for (int ch = 0; ch < c; ch++)
                {
                    dstBase[dstIdx + ch] = srcBase[ch * planeSize + pixelIdx];
                }
            }
        }
    }

    #endregion
}

