using System;
using System.Diagnostics;
using TokenVector.Vision.Common;
using TokenVector.Vision.Transforms;
using Xunit;

namespace TokenVector.Vision.Tests;

public class ZeroAllocationTests
{
    [Fact]
    public void FusedResizeNormalizeCHW_Preallocated_ZeroGCAllocations()
    {
        using var src = ImageBuffer.CreateRgb(224, 224);
        using var dst = ImageBuffer.CreateFloat32CHW(112, 112, 3);

        float[] mean = [0.485f, 0.456f, 0.406f];
        float[] std = [0.229f, 0.224f, 0.225f];

        // Warm up JIT and cache
        FusedTransforms.ResizeNormalizeCHW(src, 112, 112, mean, std, dst);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();

        // Hot Path Execution
        for (int i = 0; i < 10; i++)
        {
            FusedTransforms.ResizeNormalizeCHW(src, 112, 112, mean, std, dst);
        }

        long allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
        long totalAllocated = allocatedAfter - allocatedBefore;

        Assert.Equal(0, totalAllocated);
    }
}
