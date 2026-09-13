using System;
using System.IO;
using TokenVector.Vision.Codecs;
using TokenVector.Vision.Common;
using Xunit;

namespace TokenVector.Vision.Tests;

public class CodecTests
{
    [Fact]
    public void BMP_EncodeAndDecode_PreservesPixelData()
    {
        using var original = ImageBuffer.CreateRgb(16, 16);
        for (int y = 0; y < 16; y++)
        {
            var row = original.GetRowByteSpan(y);
            for (int x = 0; x < 16; x++)
            {
                row[x * 3] = (byte)(x * 10);
                row[x * 3 + 1] = (byte)(y * 10);
                row[x * 3 + 2] = (byte)((x + y) * 5);
            }
        }

        byte[] bmpBytes = ImageEncoder.Instance.EncodeToBytes(original, "bmp");
        Assert.NotNull(bmpBytes);
        Assert.True(bmpBytes.Length > 54);

        // Header check
        var info = ImageDecoder.Instance.DecodeHeader(bmpBytes);
        Assert.Equal(16, info.Width);
        Assert.Equal(16, info.Height);
        Assert.Equal(3, info.Channels);

        // Full decode
        using var decoded = ImageDecoder.Instance.Decode(bmpBytes);
        Assert.Equal(16, decoded.Width);
        Assert.Equal(16, decoded.Height);

        for (int y = 0; y < 16; y++)
        {
            var origRow = original.GetRowByteSpan(y);
            var decRow = decoded.GetRowByteSpan(y);
            for (int x = 0; x < 16 * 3; x++)
            {
                Assert.Equal(origRow[x], decRow[x]);
            }
        }
    }

    [Fact]
    public void QOI_EncodeAndDecode_LosslessReconstruction()
    {
        using var original = ImageBuffer.CreateRgb(32, 32);
        original.AsByteSpan().Fill(120);

        byte[] qoiBytes = ImageEncoder.Instance.EncodeToBytes(original, "qoi");
        Assert.NotNull(qoiBytes);

        var info = ImageDecoder.Instance.DecodeHeader(qoiBytes);
        Assert.Equal(32, info.Width);
        Assert.Equal(32, info.Height);
        Assert.Equal(3, info.Channels);

        using var decoded = ImageDecoder.Instance.Decode(qoiBytes);
        Assert.Equal(32, decoded.Width);
        Assert.Equal(32, decoded.Height);
        Assert.Equal(120, decoded.AsByteSpan()[0]);
    }
}
