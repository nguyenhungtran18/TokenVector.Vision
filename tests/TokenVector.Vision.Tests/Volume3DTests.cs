using System;
using TokenVector.Vision.Common;
using TokenVector.Vision.Detection;
using TokenVector.Vision.Volumetric;
using Xunit;

namespace TokenVector.Vision.Tests;

public class Volume3DTests
{
    [Fact]
    public void Volume3DBuffer_CreationAndAccess_CorrectValues()
    {
        using var volume = Volume3DBuffer.CreateFloat32(16, 32, 32, channels: 1);
        Assert.Equal(16, volume.Depth);
        Assert.Equal(32, volume.Height);
        Assert.Equal(32, volume.Width);
        Assert.Equal(1, volume.Channels);
        Assert.True(volume.IsFloatingPoint);
        Assert.Equal(16 * 32 * 32 * sizeof(float), volume.TotalBytes);

        volume.SetVoxelFloat(5, 10, 15, 42.5f);
        Assert.Equal(42.5f, volume.GetVoxelFloat(5, 10, 15));
    }

    [Fact]
    public void Volume3DBuffer_GetSlice_ZeroCopy2DImage()
    {
        using var volume = Volume3DBuffer.CreateFloat32(10, 20, 20, channels: 1);
        volume.SetVoxelFloat(3, 5, 5, 99.0f);

        // Get 2D slice at z = 3
        using var slice = volume.GetSlice(3);
        Assert.Equal(20, slice.Width);
        Assert.Equal(20, slice.Height);
        Assert.Equal(PixelFormat.Float32Grayscale, slice.Format);

        unsafe
        {
            float val = slice.FloatPointer[(5 * 20) + 5];
            Assert.Equal(99.0f, val);
        }
    }

    [Fact]
    public void Volume3DBuffer_SubvolumeExtraction_CorrectBounds()
    {
        using var volume = Volume3DBuffer.CreateFloat32(32, 32, 32);
        volume.Fill(1.0f);
        volume.SetVoxelFloat(10, 10, 10, 77.0f);

        using var sub = volume.ExtractSubvolume(8, 8, 8, 8, 8, 8);
        Assert.Equal(8, sub.Depth);
        Assert.Equal(8, sub.Height);
        Assert.Equal(8, sub.Width);
        Assert.Equal(77.0f, sub.GetVoxelFloat(2, 2, 2));
    }

    [Fact]
    public void Volume3DTransforms_TrilinearResample_AccurateInterpolation()
    {
        // 4x4x4 volume with corners
        using var src = Volume3DBuffer.CreateFloat32(4, 4, 4);
        src.Fill(0.0f);
        src.SetVoxelFloat(0, 0, 0, 100.0f);

        using var dst = Volume3DTransforms.ResampleTrilinear(src, 8, 8, 8);
        Assert.Equal(8, dst.Depth);
        Assert.Equal(8, dst.Height);
        Assert.Equal(8, dst.Width);
        Assert.True(dst.GetVoxelFloat(0, 0, 0) > 0.0f);
    }

    [Fact]
    public void Volume3DTransforms_BoxFilter_SmoothesVolume()
    {
        using var src = Volume3DBuffer.CreateFloat32(10, 10, 10);
        src.Fill(0.0f);
        src.SetVoxelFloat(5, 5, 5, 270.0f);

        using var filtered = Volume3DTransforms.FilterBox3D(src);
        Assert.Equal(10.0f, filtered.GetVoxelFloat(5, 5, 5));
    }

    [Fact]
    public void Volume3DTransforms_MaxPooling_Downsamples2x()
    {
        using var src = Volume3DBuffer.CreateFloat32(8, 8, 8);
        src.Fill(0.0f);
        src.SetVoxelFloat(0, 0, 0, 55.0f);

        using var pooled = Volume3DTransforms.MaxPooling3D(src);
        Assert.Equal(4, pooled.Depth);
        Assert.Equal(4, pooled.Height);
        Assert.Equal(4, pooled.Width);
        Assert.Equal(55.0f, pooled.GetVoxelFloat(0, 0, 0));
    }

    [Fact]
    public void BoundingBox3D_IoU_AccurateComputation()
    {
        var box1 = new BoundingBox3D(0, 0, 0, 10, 10, 10); // Vol = 1000
        var box2 = new BoundingBox3D(5, 0, 0, 15, 10, 10); // Vol = 1000, Inter = 5*10*10 = 500
        // Union = 1000 + 1000 - 500 = 1500 -> IoU = 500 / 1500 = 0.3333f

        float iou = box1.IoU(box2);
        Assert.InRange(iou, 0.33f, 0.34f);

        // Identical box
        Assert.Equal(1.0f, box1.IoU(box1));
    }
}
