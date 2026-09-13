using System;
using TokenVector.Vision.Batching;
using TokenVector.Vision.Common;
using TokenVector.Vision.Detection;
using TokenVector.Vision.Drawing;
using TokenVector.Vision.Transforms;
using Xunit;


namespace TokenVector.Vision.Tests;

public class DetectionTests
{
    [Fact]
    public void BoundingBox_IoU_ComputesCorrectly()
    {
        // Box 1: [0, 0, 10, 10] -> Area = 100
        var boxA = new BoundingBox(0, 0, 10, 10, score: 0.9f, classId: 0);
        // Box 2: [5, 0, 15, 10] -> Intersection = 5 * 10 = 50, Union = 100 + 100 - 50 = 150 -> IoU = 50/150 = 0.3333f
        var boxB = new BoundingBox(5, 0, 15, 10, score: 0.8f, classId: 0);

        float iou = boxA.ComputeIoU(boxB);
        Assert.Equal(50.0f / 150.0f, iou, precision: 4);

        // Disjoint boxes -> IoU = 0
        var boxC = new BoundingBox(20, 20, 30, 30);
        Assert.Equal(0.0f, boxA.ComputeIoU(boxC));

        // Exact same box -> IoU = 1.0
        Assert.Equal(1.0f, boxA.ComputeIoU(boxA), precision: 5);
    }

    [Fact]
    public void LetterboxTransform_PreservesAspectRatio_And_InvertsCoordinates()
    {
        // 1920x1080 -> 640x640
        using var src = ImageBuffer.CreateRgb(1920, 1080);
        src.AsByteSpan().Fill(200);

        using var letterboxed = LetterboxTransform.Apply(src, 640, 640, padValue: 114, out var metadata);

        Assert.Equal(640, letterboxed.Width);
        Assert.Equal(640, letterboxed.Height);
        Assert.Equal(640.0f / 1920.0f, metadata.Scale, precision: 4);
        Assert.Equal(0, metadata.PadX);
        Assert.Equal((640 - (int)MathF.Round(1080 * metadata.Scale)) / 2, metadata.PadY);

        // Test Inverse Coordinate Mapping
        // Suppose a detection occurs at the center of the letterboxed image
        var detBox = new BoundingBox(metadata.PadX + 100, metadata.PadY + 50, metadata.PadX + 300, metadata.PadY + 200);
        var origBox = metadata.InverseTransform(detBox);

        Assert.True(origBox.X1 >= 0 && origBox.X2 <= 1920);
        Assert.True(origBox.Y1 >= 0 && origBox.Y2 <= 1080);
        Assert.Equal(100 / metadata.Scale, origBox.X1, precision: 2);
    }

    [Fact]
    public void NonMaximumSuppression_SuppressesOverlappingBoxes()
    {
        var candidates = new BoundingBox[]
        {
            new(10, 10, 50, 50, score: 0.95f, classId: 0), // Base top score
            new(12, 11, 52, 49, score: 0.90f, classId: 0), // Overlaps base heavily (IoU > 0.8) -> Should be suppressed
            new(100, 100, 150, 150, score: 0.85f, classId: 0), // Separate object -> Keep
            new(10, 10, 50, 50, score: 0.80f, classId: 1), // Overlaps base but DIFFERENT class (1 vs 0) -> Keep
            new(5, 5, 20, 20, score: 0.10f, classId: 0) // Below score threshold (0.25) -> Should be filtered
        };

        var filtered = NonMaximumSuppression.Filter(candidates, iouThreshold: 0.5f, scoreThreshold: 0.25f);

        Assert.Equal(3, filtered.Length);
        Assert.Equal(0.95f, filtered[0].Score);
        Assert.Equal(0.85f, filtered[1].Score);
        Assert.Equal(0.80f, filtered[2].Score);
        Assert.Equal(1, filtered[2].ClassId);
    }

    [Fact]
    public void BoundingBoxTransforms_HorizontalFlip_Correct()
    {
        int imgW = 640;
        var box = new BoundingBox(100, 50, 200, 150);
        var flipped = BoundingBoxTransforms.HorizontalFlip(box, imgW);

        Assert.Equal(imgW - 200, flipped.X1); // 440
        Assert.Equal(imgW - 100, flipped.X2); // 540
        Assert.Equal(50, flipped.Y1);
        Assert.Equal(150, flipped.Y2);
    }

    [Fact]
    public unsafe void ImagePainter_DrawsBoundingBox_WithoutErrors()
    {
        using var img = ImageBuffer.CreateRgb(640, 480);
        var box = new BoundingBox(50, 50, 200, 180);

        ImagePainter.DrawBoundingBox(img, box, label: "PERSON: 95%", color: RgbColor.Green, thickness: 2, fillAlpha: 0.2f);
        ImagePainter.FillCircle(img, 125, 115, radius: 5, color: RgbColor.Red);
        ImagePainter.DrawLine(img, 0, 0, 100, 100, color: RgbColor.Yellow, thickness: 2);

        // Verify some pixels were written
        byte* ptr = img.BytePointer;
        // Verify center pixel of the circle (125, 115) is Red
        byte* circleCenter = ptr + 115 * img.Stride + 125 * 3;
        Assert.Equal(255, circleCenter[0]); // R
        Assert.Equal(0, circleCenter[1]);   // G
        Assert.Equal(0, circleCenter[2]);   // B
    }

    [Fact]
    public unsafe void BatchOps_StackNCHW_CreatesContiguousTensor()
    {
        using var img1 = ImageBuffer.CreateFloat32CHW(32, 32, 3);
        using var img2 = ImageBuffer.CreateFloat32CHW(32, 32, 3);
        using var img3 = ImageBuffer.CreateFloat32CHW(32, 32, 3);
        using var img4 = ImageBuffer.CreateFloat32CHW(32, 32, 3);

        img1.GetChannelPlaneFloatSpan(0).Fill(1.0f);
        img2.GetChannelPlaneFloatSpan(0).Fill(2.0f);
        img3.GetChannelPlaneFloatSpan(0).Fill(3.0f);
        img4.GetChannelPlaneFloatSpan(0).Fill(4.0f);

        ImageBuffer[] batch = [img1, img2, img3, img4];
        using var stacked = BatchOps.StackNCHW(batch);

        Assert.Equal(MemoryLayout.NCHW, stacked.Layout);
        Assert.Equal(32, stacked.Width);
        Assert.Equal(32 * 4, stacked.Height); // Combined height for NCHW

        float* ptr = stacked.FloatPointer;
        int singleImageFloats = 3 * 32 * 32;

        Assert.Equal(1.0f, ptr[0]);
        Assert.Equal(2.0f, ptr[singleImageFloats]);
        Assert.Equal(3.0f, ptr[singleImageFloats * 2]);
        Assert.Equal(4.0f, ptr[singleImageFloats * 3]);
    }

    [Fact]
    public unsafe void FusedTransforms_LetterboxNormalizeCHW_ExecutesCorrectly()
    {
        using var src = ImageBuffer.CreateRgb(1920, 1080);
        src.AsByteSpan().Fill(128);

        float[] mean = [0.485f, 0.456f, 0.406f];
        float[] std = [0.229f, 0.224f, 0.225f];

        using var tensor = FusedTransforms.LetterboxNormalizeCHW(src, 640, 640, mean, std, padValue: 114, out var meta);

        Assert.Equal(640, tensor.Width);
        Assert.Equal(640, tensor.Height);
        Assert.Equal(MemoryLayout.CHW, tensor.Layout);

        // Check padded region (top row 0,0)
        float* plane0 = tensor.GetChannelPlaneFloatPointer(0);
        float expectedPad0 = ((114.0f / 255.0f) - mean[0]) / std[0];
        Assert.Equal(expectedPad0, plane0[0], precision: 3);
    }

    [Fact]
    public void VisionBufferPool_RentsAndReturnsCorrectly()
    {
        var info = ImageInfo.Rgb(640, 480);
        using var pool = new VisionBufferPool(info, initialPrewarm: 2, maxCapacity: 4);

        Assert.Equal(2, pool.AvailableCount);

        var buf1 = pool.Rent();
        var buf2 = pool.Rent();
        Assert.Equal(0, pool.AvailableCount);

        var buf3 = pool.Rent(); // allocates on-demand
        Assert.Equal(3, pool.TotalAllocated);

        pool.Return(buf1);
        pool.Return(buf2);
        pool.Return(buf3);
        Assert.Equal(3, pool.AvailableCount);
    }
}


