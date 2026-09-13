using System;
using TokenVector.Vision.Common;
using TokenVector.Vision.Filters;
using Xunit;

namespace TokenVector.Vision.Tests;

public class FiltersAndCannyTests
{
    [Fact]
    public void GaussianBlur_SmoothsImageCorrectly()
    {
        using var src = ImageBuffer.CreateRgb(50, 50);
        src.AsByteSpan().Fill(100);
        src.GetRowByteSpan(25)[25 * 3] = 255; // Impulse noise

        var blur = new GaussianBlur(sigma: 1.5f);
        using var dst = blur.Apply(src);

        Assert.Equal(50, dst.Width);
        Assert.Equal(50, dst.Height);
        // Impulse pixel should be dispersed
        Assert.True(dst.GetRowByteSpan(25)[25 * 3] < 255);
    }

    [Fact]
    public void Sobel_DetectsHighContrastEdges()
    {
        using var src = ImageBuffer.CreateGrayscale(20, 20);
        src.Clear();
        // Create strong vertical edge at x=10
        for (int y = 0; y < 20; y++)
        {
            var row = src.GetRowByteSpan(y);
            for (int x = 10; x < 20; x++) row[x] = 255;
        }

        using var edge = EdgeDetection.Sobel(src);
        Assert.True(edge.GetRowByteSpan(10)[10] > 100); // Strong edge at boundary
        Assert.True(edge.GetRowByteSpan(10)[0] == 0);   // Flat region has 0 edge
    }

    [Fact]
    public void CannyDetector_OutputsBinaryEdges()
    {
        using var src = ImageBuffer.CreateGrayscale(30, 30);
        src.Clear();
        for (int y = 5; y < 25; y++)
        {
            var row = src.GetRowByteSpan(y);
            for (int x = 5; x < 25; x++) row[x] = 255;
        }

        var canny = new CannyDetector(lowThreshold: 30, highThreshold: 100);
        using var edges = canny.Apply(src);

        Assert.Equal(30, edges.Width);
        Assert.Equal(30, edges.Height);
        Assert.Equal(1, edges.Channels);
    }
}
