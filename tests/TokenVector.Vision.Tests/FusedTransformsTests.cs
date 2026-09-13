using System;
using TokenVector.Vision.Common;
using TokenVector.Vision.Transforms;
using Xunit;

namespace TokenVector.Vision.Tests;

public unsafe class FusedTransformsTests
{
    [Fact]
    public void Fused_ResizeNormalizeCHW_MatchesSequentialPipeline()
    {
        int srcW = 120, srcH = 120;
        int dstW = 60, dstH = 60;

        using var src = ImageBuffer.CreateRgb(srcW, srcH);
        for (int y = 0; y < srcH; y++)
        {
            var row = src.GetRowByteSpan(y);
            for (int x = 0; x < srcW; x++)
            {
                row[x * 3] = (byte)(x * 2);
                row[x * 3 + 1] = (byte)(y * 2);
                row[x * 3 + 2] = 128;
            }
        }

        float[] mean = [0.485f, 0.456f, 0.406f];
        float[] std = [0.229f, 0.224f, 0.225f];

        // 1. Single Pass Fused Operator
        using var fused = FusedTransforms.ResizeNormalizeCHW(src, dstW, dstH, mean, std);

        // 2. Sequential 3-Pass Execution
        var resize = new ResizeTransform(dstW, dstH, InterpolationMode.Bilinear);
        using var resized = resize.Apply(src);

        var norm = new NormalizeTransform(mean, std);
        using var normalized = norm.Apply(resized);

        var layout = LayoutTransforms.ToCHW;
        using var sequential = layout.Apply(normalized);

        // Compare planar results
        Assert.Equal(MemoryLayout.CHW, fused.Layout);
        Assert.Equal(dstW, fused.Width);
        Assert.Equal(dstH, fused.Height);
        Assert.Equal(3, fused.Channels);

        int total = dstW * dstH * 3;
        float* fPtr = fused.FloatPointer;
        float* sPtr = sequential.FloatPointer;

        float maxDiff = 0.0f;
        for (int i = 0; i < total; i++)
        {
            float diff = MathF.Abs(fPtr[i] - sPtr[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        // Numerical tolerance should be virtually identical (< 0.05)
        Assert.True(maxDiff < 0.05f, $"Max difference was {maxDiff}");
    }
}

