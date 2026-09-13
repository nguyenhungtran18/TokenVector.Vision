using System;
using TokenVector.Vision.Common;
using TokenVector.Vision.Transforms;
using Xunit;

namespace TokenVector.Vision.Tests;

public unsafe class LayoutTransformsTests
{
    [Fact]
    public void Transpose_HwcToChw_AndBack_PreservesExactValues()
    {
        using var hwc = ImageBuffer.CreateFloat32HWC(10, 10, 3);
        float* ptr = hwc.FloatPointer;
        for (int i = 0; i < 300; i++) ptr[i] = i * 0.5f;

        using var chw = LayoutTransforms.ToCHW.Apply(hwc);
        Assert.Equal(MemoryLayout.CHW, chw.Layout);

        using var restoredHwc = LayoutTransforms.ToHWC.Apply(chw);
        Assert.Equal(MemoryLayout.HWC, restoredHwc.Layout);

        float* restPtr = restoredHwc.FloatPointer;
        for (int i = 0; i < 300; i++)
        {
            Assert.Equal(ptr[i], restPtr[i]);
        }
    }
}

