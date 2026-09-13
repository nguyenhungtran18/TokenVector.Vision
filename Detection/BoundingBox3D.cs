using System;
using System.Runtime.CompilerServices;

namespace TokenVector.Vision.Detection;

/// <summary>
/// High-performance struct for 3D Axis-Aligned and Volumetric Bounding Boxes.
/// Used in 3D Object Detection, LiDAR perception, and Medical CT/MRI lesion localization.
/// </summary>
public readonly struct BoundingBox3D : IEquatable<BoundingBox3D>
{
    public float MinX { get; }
    public float MinY { get; }
    public float MinZ { get; }
    public float MaxX { get; }
    public float MaxY { get; }
    public float MaxZ { get; }
    public float Confidence { get; }
    public int ClassId { get; }

    public float SizeX => MathF.Max(0.0f, MaxX - MinX);
    public float SizeY => MathF.Max(0.0f, MaxY - MinY);
    public float SizeZ => MathF.Max(0.0f, MaxZ - MinZ);
    public float Volume => SizeX * SizeY * SizeZ;

    public float CenterX => (MinX + MaxX) * 0.5f;
    public float CenterY => (MinY + MaxY) * 0.5f;
    public float CenterZ => (MinZ + MaxZ) * 0.5f;

    public BoundingBox3D(float minX, float minY, float minZ, float maxX, float maxY, float maxZ, float confidence = 1.0f, int classId = 0)
    {
        MinX = MathF.Min(minX, maxX);
        MinY = MathF.Min(minY, maxY);
        MinZ = MathF.Min(minZ, maxZ);
        MaxX = MathF.Max(minX, maxX);
        MaxY = MathF.Max(minY, maxY);
        MaxZ = MathF.Max(minZ, maxZ);
        Confidence = confidence;
        ClassId = classId;
    }

    public static BoundingBox3D FromCenterSize(float cx, float cy, float cz, float sx, float sy, float sz, float confidence = 1.0f, int classId = 0)
    {
        float hx = sx * 0.5f;
        float hy = sy * 0.5f;
        float hz = sz * 0.5f;
        return new BoundingBox3D(cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz, confidence, classId);
    }

    /// <summary>
    /// Computes the 3D Intersection over Union (IoU) between two 3D bounding boxes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float IoU(in BoundingBox3D other)
    {
        float interMinX = MathF.Max(MinX, other.MinX);
        float interMinY = MathF.Max(MinY, other.MinY);
        float interMinZ = MathF.Max(MinZ, other.MinZ);

        float interMaxX = MathF.Min(MaxX, other.MaxX);
        float interMaxY = MathF.Min(MaxY, other.MaxY);
        float interMaxZ = MathF.Min(MaxZ, other.MaxZ);

        float interSizeX = MathF.Max(0.0f, interMaxX - interMinX);
        float interSizeY = MathF.Max(0.0f, interMaxY - interMinY);
        float interSizeZ = MathF.Max(0.0f, interMaxZ - interMinZ);

        float intersectionVolume = interSizeX * interSizeY * interSizeZ;
        if (intersectionVolume <= 0.0f)
            return 0.0f;

        float unionVolume = Volume + other.Volume - intersectionVolume;
        return unionVolume > 0.0f ? intersectionVolume / unionVolume : 0.0f;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Equals(BoundingBox3D other) =>
        MinX == other.MinX && MinY == other.MinY && MinZ == other.MinZ &&
        MaxX == other.MaxX && MaxY == other.MaxY && MaxZ == other.MaxZ &&
        Confidence == other.Confidence && ClassId == other.ClassId;

    public override bool Equals(object? obj) => obj is BoundingBox3D other && Equals(other);

    public override int GetHashCode() => HashCode.Combine(MinX, MinY, MinZ, MaxX, MaxY, MaxZ);

    public override string ToString() => $"BoundingBox3D([{MinX:F1}, {MinY:F1}, {MinZ:F1}] -> [{MaxX:F1}, {MaxY:F1}, {MaxZ:F1}], Vol={Volume:F1}, Conf={Confidence:F2})";
}
