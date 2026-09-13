using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace TokenVector.Vision.Common;

/// <summary>
/// High-performance unmanaged image buffer providing Zero-GC memory management,
/// SIMD 64-byte alignment, sub-region zero-copy slicing, and fast pointer access.
/// </summary>
public sealed unsafe class ImageBuffer : IDisposable
{
    private const nuint Alignment = 64; // 64-byte alignment for AVX-512 and cacheline alignment

    private readonly void* _rawPointer;
    private readonly ImageInfo _info;
    private readonly bool _isOwner;
    private readonly ImageBuffer? _parentBuffer;
    private readonly int _byteOffset;
    private int _refCount;
    private int _isDisposed;

    /// <summary>
    /// Image dimensions and format descriptor.
    /// </summary>
    public ImageInfo Info => _info;

    /// <summary>
    /// Width in pixels.
    /// </summary>
    public int Width => _info.Width;

    /// <summary>
    /// Height in pixels.
    /// </summary>
    public int Height => _info.Height;

    /// <summary>
    /// Number of color channels.
    /// </summary>
    public int Channels => _info.Channels;

    /// <summary>
    /// Pixel format.
    /// </summary>
    public PixelFormat Format => _info.Format;

    /// <summary>
    /// Row stride in bytes.
    /// </summary>
    public int Stride => _info.Stride;

    /// <summary>
    /// Memory layout (HWC, CHW, etc.).
    /// </summary>
    public MemoryLayout Layout => _info.Layout;

    /// <summary>
    /// Whether data is 32-bit floating point.
    /// </summary>
    public bool IsFloatingPoint => _info.IsFloatingPoint;

    /// <summary>
    /// Total buffer size in bytes.
    /// </summary>
    public int ByteLength => _info.TotalBytes;

    /// <summary>
    /// Total element count (pixels * channels).
    /// </summary>
    public int ElementCount => _info.TotalElements;

    /// <summary>
    /// Whether this buffer has been disposed.
    /// </summary>
    public bool IsDisposed => Volatile.Read(ref _isDisposed) != 0;

    /// <summary>
    /// Raw unmanaged pointer to the first byte of pixel data.
    /// </summary>
    public byte* BytePointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if (IsDisposed) throw new ObjectDisposedException(nameof(ImageBuffer));
            return (byte*)_rawPointer + _byteOffset;
        }
    }

    /// <summary>
    /// Raw unmanaged pointer to 32-bit floating point pixel data.
    /// </summary>
    public float* FloatPointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if (IsDisposed) throw new ObjectDisposedException(nameof(ImageBuffer));
            if (!IsFloatingPoint) throw new InvalidOperationException("Buffer is not floating point.");
            return (float*)((byte*)_rawPointer + _byteOffset);
        }
    }

    #region Constructors and Factories

    /// <summary>
    /// Allocates an aligned unmanaged memory buffer for the specified image metadata.
    /// </summary>
    public ImageBuffer(ImageInfo info)
    {
        _info = info;
        nuint byteCount = (nuint)info.TotalBytes;
        if (byteCount == 0) byteCount = 64; // minimum allocation

        _rawPointer = NativeMemory.AlignedAlloc(byteCount, Alignment);
        NativeMemory.Clear(_rawPointer, byteCount);

        _byteOffset = 0;
        _isOwner = true;
        _parentBuffer = null;
        _refCount = 1;
        _isDisposed = 0;
    }

    /// <summary>
    /// Wraps an existing unmanaged memory block without copy.
    /// </summary>
    public ImageBuffer(void* unmanagedPointer, ImageInfo info, bool isOwner = false)
    {
        if (unmanagedPointer == null) throw new ArgumentNullException(nameof(unmanagedPointer));
        _rawPointer = unmanagedPointer;
        _info = info;
        _byteOffset = 0;
        _isOwner = isOwner;
        _parentBuffer = null;
        _refCount = isOwner ? 1 : 0;
        _isDisposed = 0;
    }

    /// <summary>
    /// Internal view constructor for sub-region slicing.
    /// </summary>
    private ImageBuffer(ImageBuffer parent, int byteOffset, ImageInfo subInfo)
    {
        _parentBuffer = parent ?? throw new ArgumentNullException(nameof(parent));
        _parentBuffer.AddRef();
        _rawPointer = parent._rawPointer;
        _byteOffset = parent._byteOffset + byteOffset;
        _info = subInfo;
        _isOwner = false;
        _refCount = 1;
        _isDisposed = 0;
    }

    /// <summary>
    /// Creates a newly allocated RGB24 image buffer.
    /// </summary>
    public static ImageBuffer CreateRgb(int width, int height) => new(ImageInfo.Rgb(width, height));

    /// <summary>
    /// Creates a newly allocated Grayscale8 image buffer.
    /// </summary>
    public static ImageBuffer CreateGrayscale(int width, int height) => new(ImageInfo.Grayscale(width, height));

    /// <summary>
    /// Creates a newly allocated RGBA32 image buffer.
    /// </summary>
    public static ImageBuffer CreateRgba(int width, int height) => new(ImageInfo.Rgba(width, height));

    /// <summary>
    /// Creates a newly allocated Float32 Planar (CHW) tensor buffer.
    /// </summary>
    public static ImageBuffer CreateFloat32CHW(int width, int height, int channels = 3) =>
        new(ImageInfo.Float32CHW(width, height, channels));

    /// <summary>
    /// Creates a newly allocated Float32 Batch Planar (NCHW) tensor buffer.
    /// </summary>
    public static ImageBuffer CreateFloat32NCHW(int batchSize, int channels, int height, int width) =>
        new(ImageInfo.Float32NCHW(batchSize, channels, height, width));

    /// <summary>
    /// Creates a newly allocated Float32 Interleaved (HWC) buffer.
    /// </summary>
    public static ImageBuffer CreateFloat32HWC(int width, int height, int channels = 3) =>
        new(ImageInfo.Float32HWC(width, height, channels));


    /// <summary>
    /// Creates an ImageBuffer with specific ImageInfo.
    /// </summary>
    public static ImageBuffer Create(ImageInfo info) => new(info);

    #endregion

    #region Span and Pointer Operations

    /// <summary>
    /// Returns the buffer as a byte Span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<byte> AsByteSpan()
    {
        return new Span<byte>(BytePointer, _info.TotalBytes);
    }

    /// <summary>
    /// Returns the buffer as a ReadOnlySpan of bytes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<byte> AsReadOnlyByteSpan()
    {
        return new ReadOnlySpan<byte>(BytePointer, _info.TotalBytes);
    }

    /// <summary>
    /// Returns the buffer as a float Span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> AsFloatSpan()
    {
        return new Span<float>(FloatPointer, _info.TotalElements);
    }

    /// <summary>
    /// Returns the buffer as a ReadOnlySpan of floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<float> AsReadOnlyFloatSpan()
    {
        return new ReadOnlySpan<float>(FloatPointer, _info.TotalElements);
    }

    /// <summary>
    /// Gets a pointer to the start of a specific row in HWC layout.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte* GetRowBytePointer(int y)
    {
        if ((uint)y >= (uint)_info.Height) throw new ArgumentOutOfRangeException(nameof(y));
        return BytePointer + (y * _info.Stride);
    }

    /// <summary>
    /// Gets a Span over a specific row in HWC layout.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<byte> GetRowByteSpan(int y)
    {
        int rowBytes = _info.Width * _info.Channels * (IsFloatingPoint ? sizeof(float) : sizeof(byte));
        return new Span<byte>(GetRowBytePointer(y), rowBytes);
    }

    /// <summary>
    /// Gets a pointer to a specific channel plane in CHW planar layout.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetChannelPlaneFloatPointer(int channel)
    {
        if (!IsFloatingPoint || _info.Layout != MemoryLayout.CHW)
            throw new InvalidOperationException("Operation requires Float32 planar CHW layout.");
        if ((uint)channel >= (uint)_info.Channels)
            throw new ArgumentOutOfRangeException(nameof(channel));

        int planeElements = _info.Width * _info.Height;
        return FloatPointer + (channel * planeElements);
    }

    /// <summary>
    /// Gets a Span over a specific channel plane in CHW planar layout.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> GetChannelPlaneFloatSpan(int channel)
    {
        return new Span<float>(GetChannelPlaneFloatPointer(channel), _info.Width * _info.Height);
    }

    #endregion

    #region Slicing and Views

    /// <summary>
    /// Creates a zero-copy sub-region view sharing underlying memory.
    /// Stride is preserved from the parent buffer.
    /// </summary>
    public ImageBuffer Slice(int x, int y, int width, int height)
    {
        if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
            x + width > _info.Width || y + height > _info.Height)
        {
            throw new ArgumentOutOfRangeException("Sub-region bounds exceed image dimensions.");
        }

        if (_info.Layout != MemoryLayout.HWC)
        {
            throw new NotSupportedException("Zero-copy 2D sub-region slicing is currently optimized for HWC layout.");
        }

        int bytesPerPixel = _info.Channels * (_info.IsFloatingPoint ? sizeof(float) : sizeof(byte));
        int offset = (y * _info.Stride) + (x * bytesPerPixel);

        var subInfo = new ImageInfo(width, height, _info.Channels, _info.Format, _info.Stride, _info.Layout);
        return new ImageBuffer(this, offset, subInfo);
    }

    #endregion

    #region Copy, Clone, and Memory Operations

    /// <summary>
    /// Copies pixels from this buffer into a destination buffer.
    /// </summary>
    public void CopyTo(ImageBuffer destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (destination.Width != Width || destination.Height != Height ||
            destination.Channels != Channels || destination.Format != Format)
        {
            throw new ArgumentException("Destination buffer dimensions or format mismatch.");
        }

        if (Stride == destination.Stride && Layout == destination.Layout)
        {
            Buffer.MemoryCopy(BytePointer, destination.BytePointer, destination.ByteLength, ByteLength);
        }
        else
        {
            int rowBytes = Width * Channels * (IsFloatingPoint ? sizeof(float) : sizeof(byte));
            for (int y = 0; y < Height; y++)
            {
                byte* srcRow = GetRowBytePointer(y);
                byte* dstRow = destination.GetRowBytePointer(y);
                Buffer.MemoryCopy(srcRow, dstRow, rowBytes, rowBytes);
            }
        }
    }

    /// <summary>
    /// Creates a deep detached clone with newly allocated aligned unmanaged memory.
    /// </summary>
    public ImageBuffer Clone()
    {
        var clone = new ImageBuffer(_info);
        CopyTo(clone);
        return clone;
    }

    /// <summary>
    /// Clears the entire buffer to 0 bytes.
    /// </summary>
    public void Clear()
    {
        NativeMemory.Clear(BytePointer, (nuint)_info.TotalBytes);
    }

    #endregion

    #region Reference Counting and Lifetime

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddRef()
    {
        if (IsDisposed) throw new ObjectDisposedException(nameof(ImageBuffer));
        Interlocked.Increment(ref _refCount);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (Interlocked.Exchange(ref _isDisposed, 1) != 0) return;

        if (Interlocked.Decrement(ref _refCount) <= 0)
        {
            if (_parentBuffer != null)
            {
                _parentBuffer.Dispose();
            }
            else if (_isOwner && _rawPointer != null)
            {
                NativeMemory.AlignedFree(_rawPointer);
            }
        }
    }

    ~ImageBuffer()
    {
        Dispose(false);
    }

    #endregion
}
