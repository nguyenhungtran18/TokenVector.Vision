using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Volumetric;

/// <summary>
/// High-performance unmanaged 3D Volumetric buffer for Large Spatial Dimensions (CT/MRI, 3D Vision, Voxels).
/// Supports zero-GC unmanaged memory allocation, 64-byte SIMD alignment, and 2D slice referencing.
/// </summary>
public sealed unsafe class Volume3DBuffer : IDisposable
{
    private const nuint Alignment = 64;

    private readonly void* _rawPointer;
    private readonly Volume3DInfo _info;
    private readonly bool _isOwner;
    private readonly Volume3DBuffer? _parentVolume;
    private readonly long _byteOffset;
    private int _refCount;
    private int _isDisposed;

    public Volume3DInfo Info => _info;
    public int Depth => _info.Depth;
    public int Height => _info.Height;
    public int Width => _info.Width;
    public int Channels => _info.Channels;
    public bool IsFloatingPoint => _info.IsFloatingPoint;
    public VolumeLayout Layout => _info.Layout;
    public long TotalBytes => _info.TotalBytes;

    public byte* BytePointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            ThrowIfDisposed();
            return (byte*)_rawPointer + _byteOffset;
        }
    }

    public float* FloatPointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            ThrowIfDisposed();
            if (!_info.IsFloatingPoint)
                throw new InvalidOperationException("Buffer is not formatted as floating-point.");
            return (float*)((byte*)_rawPointer + _byteOffset);
        }
    }

    public Volume3DBuffer(Volume3DInfo info)
    {
        _info = info;
        _byteOffset = 0;
        _isOwner = true;
        _refCount = 1;
        _parentVolume = null;

        nuint size = (nuint)info.TotalBytes;
        _rawPointer = NativeMemory.AlignedAlloc(size, Alignment);
        NativeMemory.Clear(_rawPointer, size);
    }

    private Volume3DBuffer(void* rawPointer, Volume3DInfo info, Volume3DBuffer parent, long byteOffset)
    {
        _rawPointer = rawPointer;
        _info = info;
        _isOwner = false;
        _parentVolume = parent;
        _byteOffset = byteOffset;
        _refCount = 1;
        parent.AddRef();
    }

    public static Volume3DBuffer CreateFloat32(int depth, int height, int width, int channels = 1) =>
        new(Volume3DInfo.Float32(depth, height, width, channels));

    public static Volume3DBuffer CreateByte(int depth, int height, int width, int channels = 1) =>
        new(Volume3DInfo.Byte(depth, height, width, channels));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float GetVoxelFloat(int z, int y, int x, int c = 0)
    {
        ThrowIfDisposed();
        long index = ((long)z * _info.Height * _info.Width + y * _info.Width + x) * _info.Channels + c;
        return FloatPointer[index];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SetVoxelFloat(int z, int y, int x, float val, int c = 0)
    {
        ThrowIfDisposed();
        long index = ((long)z * _info.Height * _info.Width + y * _info.Width + x) * _info.Channels + c;
        FloatPointer[index] = val;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte GetVoxelByte(int z, int y, int x, int c = 0)
    {
        ThrowIfDisposed();
        long index = ((long)z * _info.Height * _info.Width + y * _info.Width + x) * _info.Channels + c;
        return BytePointer[index];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SetVoxelByte(int z, int y, int x, byte val, int c = 0)
    {
        ThrowIfDisposed();
        long index = ((long)z * _info.Height * _info.Width + y * _info.Width + x) * _info.Channels + c;
        BytePointer[index] = val;
    }

    /// <summary>
    /// Extracts a 2D slice at depth index z as an ImageBuffer without heap memory copy.
    /// </summary>
    public ImageBuffer GetSlice(int z)
    {
        ThrowIfDisposed();
        if (z < 0 || z >= _info.Depth)
            throw new ArgumentOutOfRangeException(nameof(z), "Depth index out of range.");

        long sliceOffset = (long)z * _info.SliceStrideBytes;
        byte* slicePtr = BytePointer + sliceOffset;

        PixelFormat pixelFormat = _info.IsFloatingPoint
            ? (_info.Channels == 1 ? PixelFormat.Float32Grayscale : PixelFormat.Float32Rgb)
            : (_info.Channels == 1 ? PixelFormat.Grayscale8 : PixelFormat.Rgb24);

        var sliceInfo = new ImageInfo(_info.Width, _info.Height, _info.Channels, pixelFormat, _info.RowStrideBytes);
        return new ImageBuffer(slicePtr, sliceInfo, isOwner: false);
    }

    /// <summary>
    /// Fills the entire 3D volume with a constant float value.
    /// </summary>
    public void Fill(float value)
    {
        ThrowIfDisposed();
        if (!_info.IsFloatingPoint)
            throw new InvalidOperationException("Fill(float) is only supported for Float32 volumes.");

        float* ptr = FloatPointer;
        long total = _info.TotalElements;

        for (long i = 0; i < total; i++)
        {
            ptr[i] = value;
        }
    }

    /// <summary>
    /// Fills the entire 3D volume with a constant byte value.
    /// </summary>
    public void Fill(byte value)
    {
        ThrowIfDisposed();
        byte* ptr = BytePointer;
        long total = _info.TotalBytes;
        NativeMemory.Fill(ptr, (nuint)total, value);
    }

    /// <summary>
    /// Extracts a 3D subvolume (ROI) from this volume.
    /// </summary>
    public Volume3DBuffer ExtractSubvolume(int startZ, int startY, int startX, int depth, int height, int width)
    {
        ThrowIfDisposed();
        if (startZ < 0 || startY < 0 || startX < 0 ||
            startZ + depth > _info.Depth || startY + height > _info.Height || startX + width > _info.Width)
        {
            throw new ArgumentOutOfRangeException("Subvolume bounds exceed source volume dimensions.");
        }

        var subInfo = new Volume3DInfo(depth, height, width, _info.Channels, _info.IsFloatingPoint, _info.Layout);
        var subVolume = new Volume3DBuffer(subInfo);

        int bytesPerElem = _info.BytesPerElement;
        int rowCopyBytes = width * _info.Channels * bytesPerElem;

        byte* srcBase = BytePointer;
        byte* dstBase = subVolume.BytePointer;

        for (int z = 0; z < depth; z++)
        {
            int srcZ = startZ + z;
            long srcSliceOffset = (long)srcZ * _info.SliceStrideBytes;
            long dstSliceOffset = (long)z * subInfo.SliceStrideBytes;

            for (int y = 0; y < height; y++)
            {
                int srcY = startY + y;
                long srcOffset = srcSliceOffset + (srcY * _info.RowStrideBytes) + (startX * _info.Channels * bytesPerElem);
                long dstOffset = dstSliceOffset + (y * subInfo.RowStrideBytes);

                Buffer.MemoryCopy(srcBase + srcOffset, dstBase + dstOffset, rowCopyBytes, rowCopyBytes);
            }
        }

        return subVolume;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void AddRef() => Interlocked.Increment(ref _refCount);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ThrowIfDisposed()
    {
        if (_isDisposed != 0)
            throw new ObjectDisposedException(nameof(Volume3DBuffer));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _isDisposed, 1) != 0)
            return;

        if (Interlocked.Decrement(ref _refCount) == 0)
        {
            if (_isOwner && _rawPointer != null)
            {
                NativeMemory.AlignedFree(_rawPointer);
            }
            _parentVolume?.Dispose();
        }
    }
}
