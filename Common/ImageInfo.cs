using System;
using System.Runtime.CompilerServices;

namespace TokenVector.Vision.Common;

/// <summary>
/// Comprehensive metadata descriptor for image buffers and tensors.
/// </summary>
public readonly struct ImageInfo : IEquatable<ImageInfo>
{
    public int Width { get; }
    public int Height { get; }
    public int Channels { get; }
    public PixelFormat Format { get; }
    public int Stride { get; }
    public MemoryLayout Layout { get; }
    public bool IsFloatingPoint { get; }

    public int TotalPixels => Width * Height;
    public int TotalElements => Width * Height * Channels;
    public int BytesPerElement => IsFloatingPoint ? sizeof(float) : sizeof(byte);
    public int TotalBytes => Layout == MemoryLayout.HWC ? Stride * Height : TotalElements * BytesPerElement;

    public ImageInfo(int width, int height, int channels, PixelFormat format, int stride = 0, MemoryLayout layout = MemoryLayout.HWC)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "Width must be positive.");
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "Height must be positive.");
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels), "Channels must be positive.");

        Width = width;
        Height = height;
        Channels = channels;
        Format = format;
        Layout = layout;

        IsFloatingPoint = format is PixelFormat.Float32Grayscale or PixelFormat.Float32Rgb 
                                or PixelFormat.Float32Planar or PixelFormat.Float32Rgba;

        int elementSize = IsFloatingPoint ? sizeof(float) : sizeof(byte);
        int defaultRowBytes = width * channels * elementSize;
        Stride = stride > 0 ? stride : (layout == MemoryLayout.HWC ? defaultRowBytes : width * elementSize);
    }

    public static ImageInfo Rgb(int width, int height) =>
        new(width, height, 3, PixelFormat.Rgb24, width * 3, MemoryLayout.HWC);

    public static ImageInfo Grayscale(int width, int height) =>
        new(width, height, 1, PixelFormat.Grayscale8, width, MemoryLayout.HWC);

    public static ImageInfo Rgba(int width, int height) =>
        new(width, height, 4, PixelFormat.Rgba32, width * 4, MemoryLayout.HWC);

    public static ImageInfo Float32CHW(int width, int height, int channels = 3) =>
        new(width, height, channels, PixelFormat.Float32Planar, width * sizeof(float), MemoryLayout.CHW);

    public static ImageInfo Float32NCHW(int batchSize, int channels, int height, int width) =>
        new(width, height * batchSize, channels, PixelFormat.Float32Planar, width * sizeof(float), MemoryLayout.NCHW);

    public static ImageInfo Float32HWC(int width, int height, int channels = 3) =>
        new(width, height, channels, PixelFormat.Float32Rgb, width * channels * sizeof(float), MemoryLayout.HWC);


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Equals(ImageInfo other) =>
        Width == other.Width && Height == other.Height && Channels == other.Channels &&
        Format == other.Format && Stride == other.Stride && Layout == other.Layout;

    public override bool Equals(object? obj) => obj is ImageInfo other && Equals(other);

    public override int GetHashCode() => HashCode.Combine(Width, Height, Channels, (int)Format, Stride, (int)Layout);

    public static bool operator ==(ImageInfo left, ImageInfo right) => left.Equals(right);
    public static bool operator !=(ImageInfo left, ImageInfo right) => !left.Equals(right);

    public override string ToString() => $"ImageInfo({Width}x{Height}x{Channels}, Format={Format}, Layout={Layout}, Stride={Stride})";
}
