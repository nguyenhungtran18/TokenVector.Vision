using System;
using System.Runtime.CompilerServices;

namespace TokenVector.Vision.Detection;

/// <summary>
/// Bounding box coordinate representation formats.
/// </summary>
public enum BoxFormat
{
    /// <summary>Top-Left and Bottom-Right corners: [X1, Y1, X2, Y2]</summary>
    XYXY,
    /// <summary>Top-Left corner with width and height: [X, Y, Width, Height]</summary>
    XYWH,
    /// <summary>Center point with width and height: [CenterX, CenterY, Width, Height]</summary>
    CXCYWH
}

/// <summary>
/// High-performance 2D Bounding Box representation for Object Detection and Tracking.
/// Optimized for zero-heap allocation, SIMD calculations, and fast IoU computations.
/// </summary>
public readonly record struct BoundingBox
{
    public readonly float X1;
    public readonly float Y1;
    public readonly float X2;
    public readonly float Y2;
    public readonly float Score;
    public readonly int ClassId;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public BoundingBox(float x1, float y1, float x2, float y2, float score = 1.0f, int classId = 0)
    {
        X1 = MathF.Min(x1, x2);
        Y1 = MathF.Min(y1, y2);
        X2 = MathF.Max(x1, x2);
        Y2 = MathF.Max(y1, y2);
        Score = score;
        ClassId = classId;
    }

    /// <summary>Width of the bounding box.</summary>
    public float Width => MathF.Max(0.0f, X2 - X1);

    /// <summary>Height of the bounding box.</summary>
    public float Height => MathF.Max(0.0f, Y2 - Y1);

    /// <summary>Total area of the bounding box.</summary>
    public float Area => Width * Height;

    /// <summary>Center X coordinate.</summary>
    public float CenterX => X1 + Width * 0.5f;

    /// <summary>Center Y coordinate.</summary>
    public float CenterY => Y1 + Height * 0.5f;

    /// <summary>
    /// Computes Intersection over Union (IoU) with another bounding box.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float ComputeIoU(in BoundingBox other)
    {
        float interX1 = MathF.Max(X1, other.X1);
        float interY1 = MathF.Max(Y1, other.Y1);
        float interX2 = MathF.Min(X2, other.X2);
        float interY2 = MathF.Min(Y2, other.Y2);

        float interW = MathF.Max(0.0f, interX2 - interX1);
        float interH = MathF.Max(0.0f, interY2 - interY1);
        float intersection = interW * interH;

        if (intersection <= 0.0f)
            return 0.0f;

        float union = Area + other.Area - intersection;
        return union > 0.0f ? intersection / union : 0.0f;
    }

    /// <summary>
    /// Clips the bounding box coordinates to image boundaries [0, 0, width, height].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public BoundingBox Clip(float maxWidth, float maxHeight)
    {
        float nx1 = Math.Clamp(X1, 0.0f, maxWidth);
        float ny1 = Math.Clamp(Y1, 0.0f, maxHeight);
        float nx2 = Math.Clamp(X2, 0.0f, maxWidth);
        float ny2 = Math.Clamp(Y2, 0.0f, maxHeight);
        return new BoundingBox(nx1, ny1, nx2, ny2, Score, ClassId);
    }

    /// <summary>
    /// Scales the bounding box coordinates by given scale factors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public BoundingBox Scale(float scaleX, float scaleY)
    {
        return new BoundingBox(X1 * scaleX, Y1 * scaleY, X2 * scaleX, Y2 * scaleY, Score, ClassId);
    }

    /// <summary>
    /// Shifts the bounding box coordinates by given offsets.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public BoundingBox Shift(float offsetX, float offsetY)
    {
        return new BoundingBox(X1 + offsetX, Y1 + offsetY, X2 + offsetX, Y2 + offsetY, Score, ClassId);
    }

    /// <summary>
    /// Creates a BoundingBox from arbitrary format representation.
    /// </summary>
    public static BoundingBox FromFormat(float c0, float c1, float c2, float c3, BoxFormat format, float score = 1.0f, int classId = 0)
    {
        return format switch
        {
            BoxFormat.XYXY => new BoundingBox(c0, c1, c2, c3, score, classId),
            BoxFormat.XYWH => new BoundingBox(c0, c1, c0 + c2, c1 + c3, score, classId),
            BoxFormat.CXCYWH => new BoundingBox(c0 - c2 * 0.5f, c1 - c3 * 0.5f, c0 + c2 * 0.5f, c1 + c3 * 0.5f, score, classId),
            _ => throw new ArgumentOutOfRangeException(nameof(format))
        };
    }
}
