using System;
using TokenVector.Vision.Common;
using Xunit;

namespace TokenVector.Vision.Tests;

public unsafe class ImageBufferTests
{
    [Fact]
    public void ImageBuffer_CreateRgb_HasCorrectDimensionsAndAllocations()
    {
        using var buffer = ImageBuffer.CreateRgb(100, 50);
        Assert.Equal(100, buffer.Width);
        Assert.Equal(50, buffer.Height);
        Assert.Equal(3, buffer.Channels);
        Assert.Equal(PixelFormat.Rgb24, buffer.Format);
        Assert.Equal(300, buffer.Stride);
        Assert.Equal(15000, buffer.ByteLength);
        Assert.False(buffer.IsFloatingPoint);
        Assert.False(buffer.IsDisposed);
    }

    [Fact]
    public void ImageBuffer_CreateFloat32CHW_HasCorrectPlanarLayout()
    {
        using var buffer = ImageBuffer.CreateFloat32CHW(64, 64, 3);
        Assert.Equal(64, buffer.Width);
        Assert.Equal(64, buffer.Height);
        Assert.Equal(3, buffer.Channels);
        Assert.Equal(MemoryLayout.CHW, buffer.Layout);
        Assert.True(buffer.IsFloatingPoint);
        Assert.Equal(64 * 64 * 3, buffer.ElementCount);
        Assert.Equal(64 * 64 * 3 * sizeof(float), buffer.ByteLength);
    }

    [Fact]
    public void ImageBuffer_ZeroCopySlice_SharesMemoryCorrectly()
    {
        using var buffer = ImageBuffer.CreateRgb(200, 200);
        buffer.AsByteSpan().Fill(128);

        using var slice = buffer.Slice(50, 50, 100, 100);
        Assert.Equal(100, slice.Width);
        Assert.Equal(100, slice.Height);
        Assert.Equal(200 * 3, slice.Stride); // Preserves parent stride

        // Modify slice and check parent
        slice.GetRowByteSpan(0)[0] = 255;
        Assert.Equal(255, buffer.GetRowBytePointer(50)[50 * 3]);
    }

    [Fact]
    public void ImageBuffer_CloneAndCopyTo_ProducesIndependentData()
    {
        using var original = ImageBuffer.CreateRgb(10, 10);
        original.AsByteSpan().Fill(42);

        using var clone = original.Clone();
        Assert.Equal(42, clone.AsByteSpan()[0]);

        clone.AsByteSpan()[0] = 99;
        Assert.Equal(42, original.AsByteSpan()[0]); // Original unaffected
    }
}

