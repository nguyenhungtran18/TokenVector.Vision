using System;
using System.Runtime.CompilerServices;

namespace TokenVector.Vision.Detection;

/// <summary>
/// Synchronous transformations for Bounding Boxes corresponding to Image Augmentations.
/// </summary>
public static class BoundingBoxTransforms
{
    /// <summary>
    /// Flips bounding box horizontally given the image width.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static BoundingBox HorizontalFlip(in BoundingBox box, int imageWidth)
    {
        float newX1 = imageWidth - box.X2;
        float newX2 = imageWidth - box.X1;
        return new BoundingBox(newX1, box.Y1, newX2, box.Y2, box.Score, box.ClassId);
    }

    /// <summary>
    /// Flips bounding box vertically given the image height.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static BoundingBox VerticalFlip(in BoundingBox box, int imageHeight)
    {
        float newY1 = imageHeight - box.Y2;
        float newY2 = imageHeight - box.Y1;
        return new BoundingBox(box.X1, newY1, box.X2, newY2, box.Score, box.ClassId);
    }

    /// <summary>
    /// Adjusts bounding box coordinates relative to a crop region [cropX, cropY, cropWidth, cropHeight].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static BoundingBox Crop(in BoundingBox box, int cropX, int cropY, int cropWidth, int cropHeight)
    {
        float nx1 = Math.Clamp(box.X1 - cropX, 0.0f, cropWidth);
        float ny1 = Math.Clamp(box.Y1 - cropY, 0.0f, cropHeight);
        float nx2 = Math.Clamp(box.X2 - cropX, 0.0f, cropWidth);
        float ny2 = Math.Clamp(box.Y2 - cropY, 0.0f, cropHeight);
        return new BoundingBox(nx1, ny1, nx2, ny2, box.Score, box.ClassId);
    }

    /// <summary>
    /// Transforms an array or span of bounding boxes horizontally in-place or into destination.
    /// </summary>
    public static void HorizontalFlipAll(ReadOnlySpan<BoundingBox> src, Span<BoundingBox> dst, int imageWidth)
    {
        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = HorizontalFlip(src[i], imageWidth);
        }
    }
}
