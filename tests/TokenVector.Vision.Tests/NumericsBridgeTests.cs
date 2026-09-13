using System;
using TokenVector.Numerics.Core;
using TokenVector.Vision.Common;
using TokenVector.Vision.Interop;
using Xunit;

namespace TokenVector.Vision.Tests;

public unsafe class NumericsBridgeTests
{
    [Fact]
    public void ToNDArray_ExportsCorrectPlanarTensor()
    {
        using var buffer = ImageBuffer.CreateRgb(10, 10);
        // Fill channel R=255, G=128, B=0
        for (int y = 0; y < 10; y++)
        {
            var row = buffer.GetRowByteSpan(y);
            for (int x = 0; x < 10; x++)
            {
                row[x * 3] = 255;
                row[x * 3 + 1] = 128;
                row[x * 3 + 2] = 0;
            }
        }

        using var tensor = buffer.ToNDArray(MemoryLayout.CHW);
        Assert.Equal(3, tensor.Rank);
        Assert.Equal(3, tensor.Shape[0]);
        Assert.Equal(10, tensor.Shape[1]);
        Assert.Equal(10, tensor.Shape[2]);

        // Check normalized float values [0.0, 1.0]
        Assert.Equal(1.0f, tensor[0, 0, 0], 3);
        Assert.Equal(128.0f / 255.0f, tensor[1, 0, 0], 3);
        Assert.Equal(0.0f, tensor[2, 0, 0], 3);
    }

    [Fact]
    public void AsImageBuffer_WrapsNDArrayZeroCopy()
    {
        var nativeBuf = TensorBuffer<float>.AllocateNative(3 * 20 * 20);
        using var tensor = new NDArray<float>(nativeBuf, [3, 20, 20], [400, 20, 1], 0);
        tensor[0, 5, 5] = 0.75f;

        using var imgBuffer = tensor.AsImageBuffer(MemoryLayout.CHW);
        Assert.Equal(20, imgBuffer.Width);
        Assert.Equal(20, imgBuffer.Height);
        Assert.Equal(3, imgBuffer.Channels);
        Assert.Equal(MemoryLayout.CHW, imgBuffer.Layout);

        // Check shared memory
        float* plane0 = imgBuffer.GetChannelPlaneFloatPointer(0);
        Assert.Equal(0.75f, plane0[5 * 20 + 5]);
    }
}


