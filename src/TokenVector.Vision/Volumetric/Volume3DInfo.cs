using System;
using System.Runtime.CompilerServices;

namespace TokenVector.Vision.Volumetric;

/// <summary>
/// Memory layout for 3D Volumetric images and tensors.
/// </summary>
public enum VolumeLayout
{
    /// <summary>
    /// Depth-Height-Width-Channels (Interleaved 3D Voxels: D, H, W, C)
    /// </summary>
    DHWC,

    /// <summary>
    /// Channels-Depth-Height-Width (Planar 3D AI Tensor: C, D, H, W)
    /// </summary>
    CDHW,

    /// <summary>
    /// Depth-Height-Width (Single-channel scalar volume: D, H, W)
    /// </summary>
    DHW
}

/// <summary>
/// Comprehensive metadata descriptor for 3D Volumetric buffers and tensors.
/// </summary>
public readonly struct Volume3DInfo : IEquatable<Volume3DInfo>
{
    public int Depth { get; }
    public int Height { get; }
    public int Width { get; }
    public int Channels { get; }
    public bool IsFloatingPoint { get; }
    public VolumeLayout Layout { get; }

    public int TotalVoxels => Depth * Height * Width;
    public int TotalElements => Depth * Height * Width * Channels;
    public int BytesPerElement => IsFloatingPoint ? sizeof(float) : sizeof(byte);
    public int RowStrideBytes => Width * Channels * BytesPerElement;
    public int SliceStrideBytes => Height * RowStrideBytes;
    public long TotalBytes => (long)TotalElements * BytesPerElement;

    public Volume3DInfo(int depth, int height, int width, int channels, bool isFloatingPoint, VolumeLayout layout = VolumeLayout.DHWC)
    {
        if (depth <= 0) throw new ArgumentOutOfRangeException(nameof(depth), "Depth must be positive.");
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "Height must be positive.");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "Width must be positive.");
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels), "Channels must be positive.");

        Depth = depth;
        Height = height;
        Width = width;
        Channels = channels;
        IsFloatingPoint = isFloatingPoint;
        Layout = layout;
    }

    public static Volume3DInfo Float32(int depth, int height, int width, int channels = 1, VolumeLayout layout = VolumeLayout.DHWC) =>
        new(depth, height, width, channels, isFloatingPoint: true, layout);

    public static Volume3DInfo Byte(int depth, int height, int width, int channels = 1, VolumeLayout layout = VolumeLayout.DHWC) =>
        new(depth, height, width, channels, isFloatingPoint: false, layout);

    public static Volume3DInfo CDHW(int channels, int depth, int height, int width) =>
        new(depth, height, width, channels, isFloatingPoint: true, VolumeLayout.CDHW);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Equals(Volume3DInfo other) =>
        Depth == other.Depth && Height == other.Height && Width == other.Width &&
        Channels == other.Channels && IsFloatingPoint == other.IsFloatingPoint && Layout == other.Layout;

    public override bool Equals(object? obj) => obj is Volume3DInfo other && Equals(other);

    public override int GetHashCode() => HashCode.Combine(Depth, Height, Width, Channels, IsFloatingPoint, (int)Layout);

    public static bool operator ==(Volume3DInfo left, Volume3DInfo right) => left.Equals(right);
    public static bool operator !=(Volume3DInfo left, Volume3DInfo right) => !left.Equals(right);

    public override string ToString() => $"Volume3DInfo({Depth}x{Height}x{Width}x{Channels}, Float={IsFloatingPoint}, Layout={Layout}, Size={TotalBytes / (1024.0 * 1024.0):F2}MB)";
}
