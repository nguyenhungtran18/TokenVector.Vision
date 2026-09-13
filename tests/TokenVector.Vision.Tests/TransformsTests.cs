using System;
using TokenVector.Vision.Common;
using TokenVector.Vision.Transforms;
using Xunit;

namespace TokenVector.Vision.Tests;

public unsafe class TransformsTests
{
    [Fact]
    public void Resize_Bilinear_ScalesDimensionsCorrectly()
    {
        using var src = ImageBuffer.CreateRgb(100, 100);
        src.AsByteSpan().Fill(200);

        var resize = new ResizeTransform(50, 50, InterpolationMode.Bilinear);
        using var dst = resize.Apply(src);

        Assert.Equal(50, dst.Width);
        Assert.Equal(50, dst.Height);
        Assert.Equal(200, dst.AsByteSpan()[0]);
    }

    [Fact]
    public void Resize_Bicubic_InterpolatesSmoothly()
    {
        using var src = ImageBuffer.CreateRgb(32, 32);
        src.AsByteSpan().Fill(100);

        var resize = new ResizeTransform(64, 64, InterpolationMode.Bicubic);
        using var dst = resize.Apply(src);

        Assert.Equal(64, dst.Width);
        Assert.Equal(64, dst.Height);
        Assert.Equal(100, dst.AsByteSpan()[0]);
    }

    [Fact]
    public void CenterCrop_ExtractsCorrectSubregion()
    {
        using var src = ImageBuffer.CreateRgb(100, 100);
        var crop = new CenterCropTransform(40, 40);
        using var dst = crop.Apply(src);

        Assert.Equal(40, dst.Width);
        Assert.Equal(40, dst.Height);
    }

    [Fact]
    public void HorizontalFlip_InvertsHorizontalAxis()
    {
        using var src = ImageBuffer.CreateRgb(10, 10);
        src.GetRowByteSpan(0)[0] = 255; // Leftmost pixel

        var flip = new HorizontalFlipTransform();
        using var dst = flip.Apply(src);

        Assert.Equal(255, dst.GetRowByteSpan(0)[9 * 3]); // Rightmost pixel
    }

    [Fact]
    public void Normalize_AppliesMeanAndStdVectorized()
    {
        using var src = ImageBuffer.CreateRgb(10, 10);
        src.AsByteSpan().Fill(255); // 1.0f

        float[] mean = [0.5f, 0.5f, 0.5f];
        float[] std = [0.5f, 0.5f, 0.5f];
        var norm = new NormalizeTransform(mean, std);
        using var dst = norm.Apply(src);

        // (1.0 - 0.5) / 0.5 = 1.0f
        Assert.Equal(1.0f, dst.FloatPointer[0], 3);
    }
}

