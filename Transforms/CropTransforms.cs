using System;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Transforms;

/// <summary>
/// High-performance spatial cropping and padding transforms.
/// Supports zero-copy sub-span slicing and SIMD memory copying.
/// </summary>
public sealed unsafe class CenterCropTransform : IVisionTransform
{
    public int TargetWidth { get; }
    public int TargetHeight { get; }

    public CenterCropTransform(int targetWidth, int targetHeight)
    {
        if (targetWidth <= 0) throw new ArgumentOutOfRangeException(nameof(targetWidth));
        if (targetHeight <= 0) throw new ArgumentOutOfRangeException(nameof(targetHeight));
        TargetWidth = targetWidth;
        TargetHeight = targetHeight;
    }

    public ImageInfo GetOutputInfo(ImageInfo inputInfo) =>
        new(TargetWidth, TargetHeight, inputInfo.Channels, inputInfo.Format, 0, inputInfo.Layout);

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (TargetWidth > input.Width || TargetHeight > input.Height)
        {
            throw new InvalidOperationException("Target crop size cannot be larger than input dimensions.");
        }

        int startX = (input.Width - TargetWidth) / 2;
        int startY = (input.Height - TargetHeight) / 2;

        var outInfo = GetOutputInfo(input.Info);
        var target = output ?? new ImageBuffer(outInfo);

        int bytesPerPixel = input.Channels * (input.IsFloatingPoint ? sizeof(float) : sizeof(byte));
        int copyRowBytes = TargetWidth * bytesPerPixel;

        for (int y = 0; y < TargetHeight; y++)
        {
            byte* srcRow = input.GetRowBytePointer(startY + y) + (startX * bytesPerPixel);
            byte* dstRow = target.GetRowBytePointer(y);
            Buffer.MemoryCopy(srcRow, dstRow, target.Stride, copyRowBytes);
        }

        return target;
    }
}

/// <summary>
/// Random spatial cropping transform for training data augmentation.
/// </summary>
public sealed unsafe class RandomCropTransform : IVisionTransform
{
    public int TargetWidth { get; }
    public int TargetHeight { get; }
    private readonly Random _random;

    public RandomCropTransform(int targetWidth, int targetHeight, int? seed = null)
    {
        if (targetWidth <= 0) throw new ArgumentOutOfRangeException(nameof(targetWidth));
        if (targetHeight <= 0) throw new ArgumentOutOfRangeException(nameof(targetHeight));
        TargetWidth = targetWidth;
        TargetHeight = targetHeight;
        _random = seed.HasValue ? new Random(seed.Value) : Random.Shared;
    }

    public ImageInfo GetOutputInfo(ImageInfo inputInfo) =>
        new(TargetWidth, TargetHeight, inputInfo.Channels, inputInfo.Format, 0, inputInfo.Layout);

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (TargetWidth > input.Width || TargetHeight > input.Height)
        {
            throw new InvalidOperationException("Target crop size cannot be larger than input dimensions.");
        }

        int maxOffsetX = input.Width - TargetWidth;
        int maxOffsetY = input.Height - TargetHeight;
        int startX = maxOffsetX > 0 ? _random.Next(0, maxOffsetX + 1) : 0;
        int startY = maxOffsetY > 0 ? _random.Next(0, maxOffsetY + 1) : 0;

        var outInfo = GetOutputInfo(input.Info);
        var target = output ?? new ImageBuffer(outInfo);

        int bytesPerPixel = input.Channels * (input.IsFloatingPoint ? sizeof(float) : sizeof(byte));
        int copyRowBytes = TargetWidth * bytesPerPixel;

        for (int y = 0; y < TargetHeight; y++)
        {
            byte* srcRow = input.GetRowBytePointer(startY + y) + (startX * bytesPerPixel);
            byte* dstRow = target.GetRowBytePointer(y);
            Buffer.MemoryCopy(srcRow, dstRow, target.Stride, copyRowBytes);
        }

        return target;
    }
}

/// <summary>
/// Uniform padding transform around an image.
/// </summary>
public sealed unsafe class PadTransform : IVisionTransform
{
    public int PadLeft { get; }
    public int PadTop { get; }
    public int PadRight { get; }
    public int PadBottom { get; }
    public byte FillValue { get; }

    public PadTransform(int pad, byte fillValue = 0) : this(pad, pad, pad, pad, fillValue) { }

    public PadTransform(int padLeft, int padTop, int padRight, int padBottom, byte fillValue = 0)
    {
        PadLeft = padLeft;
        PadTop = padTop;
        PadRight = padRight;
        PadBottom = padBottom;
        FillValue = fillValue;
    }

    public ImageInfo GetOutputInfo(ImageInfo inputInfo) =>
        new(inputInfo.Width + PadLeft + PadRight, inputInfo.Height + PadTop + PadBottom,
            inputInfo.Channels, inputInfo.Format, 0, inputInfo.Layout);

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var outInfo = GetOutputInfo(input.Info);
        var target = output ?? new ImageBuffer(outInfo);

        target.Clear();
        if (FillValue != 0)
        {
            target.AsByteSpan().Fill(FillValue);
        }

        int bytesPerPixel = input.Channels * (input.IsFloatingPoint ? sizeof(float) : sizeof(byte));
        int srcRowBytes = input.Width * bytesPerPixel;

        for (int y = 0; y < input.Height; y++)
        {
            byte* srcRow = input.GetRowBytePointer(y);
            byte* dstRow = target.GetRowBytePointer(PadTop + y) + (PadLeft * bytesPerPixel);
            Buffer.MemoryCopy(srcRow, dstRow, target.Stride, srcRowBytes);
        }

        return target;
    }
}
