using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Batching;

/// <summary>
/// High-throughput Batch operations for collating multiple images into an NCHW / NHWC contiguous tensor.
/// </summary>
public static unsafe class BatchOps
{
    /// <summary>
    /// Stacks an array/span of planar Float32 [C, H, W] ImageBuffers into a single contiguous NCHW unmanaged buffer.
    /// </summary>
    /// <param name="images">Array or span of planar CHW float images of identical size.</param>
    /// <returns>Contiguous unmanaged ImageBuffer containing the entire batch [N, C, H, W].</returns>
    public static ImageBuffer StackNCHW(ReadOnlySpan<ImageBuffer> images)
    {
        if (images.IsEmpty)
            throw new ArgumentException("Cannot stack an empty list of images.", nameof(images));

        int batchSize = images.Length;
        int channels = images[0].Channels;
        int height = images[0].Height;
        int width = images[0].Width;

        // Verify shape consistency
        for (int i = 1; i < batchSize; i++)
        {
            if (images[i].Width != width || images[i].Height != height || images[i].Channels != channels)
            {
                throw new InvalidOperationException($"Image at index {i} shape ({images[i].Width}x{images[i].Height}x{images[i].Channels}) does not match index 0 ({width}x{height}x{channels}).");
            }
        }

        // Allocate unified contiguous buffer: batchSize * height as virtual combined height
        // Layout: NCHW
        ImageBuffer batchBuffer = ImageBuffer.CreateFloat32NCHW(batchSize, channels, height, width);

        float* dstBase = batchBuffer.FloatPointer;
        int singleImageFloats = channels * height * width;

        for (int b = 0; b < batchSize; b++)
        {
            float* srcPtr = images[b].FloatPointer;
            float* dstPtr = dstBase + b * singleImageFloats;
            Buffer.MemoryCopy(srcPtr, dstPtr, (long)singleImageFloats * sizeof(float), (long)singleImageFloats * sizeof(float));
        }

        return batchBuffer;
    }
}
