using System;
using System.Collections.Concurrent;
using System.Threading;

namespace TokenVector.Vision.Common;

/// <summary>
/// High-throughput, Zero-Syscall Unmanaged Image Buffer Pool.
/// Reuses 64-byte aligned unmanaged ImageBuffers to eliminate OS allocation/deallocation overhead in 60-120 FPS camera pipelines.
/// </summary>
public sealed class VisionBufferPool : IDisposable
{
    private readonly ConcurrentBag<ImageBuffer> _pool;
    private readonly ImageInfo _templateInfo;
    private readonly int _maxCapacity;
    private int _currentAllocated;
    private int _isDisposed;

    /// <summary>
    /// Current number of idle reusable buffers ready in the pool.
    /// </summary>
    public int AvailableCount => _pool.Count;

    /// <summary>
    /// Total number of buffers created by this pool.
    /// </summary>
    public int TotalAllocated => _currentAllocated;

    /// <summary>
    /// Creates a new VisionBufferPool for buffers with matching dimensions and format.
    /// </summary>
    /// <param name="templateInfo">Template descriptor for buffers created by this pool.</param>
    /// <param name="initialPrewarm">Number of buffers to pre-allocate immediately.</param>
    /// <param name="maxCapacity">Maximum number of idle buffers to retain in the pool.</param>
    public VisionBufferPool(ImageInfo templateInfo, int initialPrewarm = 4, int maxCapacity = 32)
    {
        _templateInfo = templateInfo;
        _maxCapacity = maxCapacity;
        _pool = new ConcurrentBag<ImageBuffer>();

        for (int i = 0; i < initialPrewarm; i++)
        {
            _pool.Add(new ImageBuffer(_templateInfo));
            Interlocked.Increment(ref _currentAllocated);
        }
    }

    /// <summary>
    /// Rents a 64-byte aligned unmanaged ImageBuffer from the pool.
    /// If no idle buffer is available, a new one is allocated on-demand.
    /// </summary>
    public ImageBuffer Rent()
    {
        if (Volatile.Read(ref _isDisposed) != 0)
            throw new ObjectDisposedException(nameof(VisionBufferPool));

        if (_pool.TryTake(out ImageBuffer? buffer))
        {
            return buffer;
        }

        Interlocked.Increment(ref _currentAllocated);
        return new ImageBuffer(_templateInfo);
    }

    /// <summary>
    /// Returns an ImageBuffer to the pool for reuse.
    /// </summary>
    public void Return(ImageBuffer buffer)
    {
        ArgumentNullException.ThrowIfNull(buffer);

        if (Volatile.Read(ref _isDisposed) != 0 || _pool.Count >= _maxCapacity)
        {
            buffer.Dispose();
            return;
        }

        if (buffer.Info != _templateInfo)
        {
            buffer.Dispose();
            return;
        }

        _pool.Add(buffer);
    }

    /// <summary>
    /// Disposes all pre-allocated unmanaged buffers in the pool.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _isDisposed, 1) != 0)
            return;

        while (_pool.TryTake(out ImageBuffer? buffer))
        {
            buffer.Dispose();
        }
    }
}
