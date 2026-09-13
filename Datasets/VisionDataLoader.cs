using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using TokenVector.Numerics.Core;
using TokenVector.Vision.Common;
using TokenVector.Vision.Interop;
using TokenVector.Vision.Transforms;

namespace TokenVector.Vision.Datasets;

/// <summary>
/// Multi-threaded prefetching Vision Data Loader with Channel-based Double Buffering.
/// Asynchronously prepares Batch N+1 on unmanaged memory while Batch N is actively consumed.
/// </summary>
public sealed class VisionDataLoader : IAsyncDisposable
{
    public ImageFolderDataset Dataset { get; }
    public int BatchSize { get; }
    public bool Shuffle { get; }
    public bool DropLast { get; }
    public int PrefetchCount { get; }
    public int TargetWidth { get; }
    public int TargetHeight { get; }

    private readonly Channel<(NDArray<float> Images, NDArray<int> Labels)> _channel;
    private readonly CancellationTokenSource _cts;
    private Task? _producerTask;

    public VisionDataLoader(
        ImageFolderDataset dataset,
        int batchSize = 32,
        bool shuffle = true,
        bool dropLast = false,
        int prefetchCount = 4,
        int targetWidth = 224,
        int targetHeight = 224)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));

        Dataset = dataset;
        BatchSize = batchSize;
        Shuffle = shuffle;
        DropLast = dropLast;
        PrefetchCount = Math.Max(2, prefetchCount);
        TargetWidth = targetWidth;
        TargetHeight = targetHeight;

        var options = new BoundedChannelOptions(PrefetchCount)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleWriter = true,
            SingleReader = false
        };

        _channel = Channel.CreateBounded<(NDArray<float>, NDArray<int>)>(options);
        _cts = new CancellationTokenSource();
    }

    /// <summary>
    /// Starts asynchronous prefetching worker and streams batched tensors.
    /// </summary>
    public async IAsyncEnumerable<(NDArray<float> Images, NDArray<int> Labels)> GetBatchesAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(_cts.Token, cancellationToken);
        _producerTask = Task.Run(() => ProduceBatchesAsync(_channel.Writer, linkedCts.Token), linkedCts.Token);

        var reader = _channel.Reader;
        while (await reader.WaitToReadAsync(linkedCts.Token).ConfigureAwait(false))
        {
            while (reader.TryRead(out var batch))
            {
                yield return batch;
            }
        }

        await _producerTask.ConfigureAwait(false);
    }

    private async Task ProduceBatchesAsync(
        ChannelWriter<(NDArray<float> Images, NDArray<int> Labels)> writer,
        CancellationToken ct)
    {
        try
        {
            int total = Dataset.Count;
            if (total == 0)
            {
                writer.Complete();
                return;
            }

            int[] indices = new int[total];
            for (int i = 0; i < total; i++) indices[i] = i;

            if (Shuffle)
            {
                Random.Shared.Shuffle(indices);
            }

            int numBatches = total / BatchSize;
            if (!DropLast && (total % BatchSize != 0)) numBatches++;

            int singleImageElements = 3 * TargetHeight * TargetWidth;

            for (int b = 0; b < numBatches; b++)
            {
                ct.ThrowIfCancellationRequested();

                int currentBatchSize = Math.Min(BatchSize, total - (b * BatchSize));
                if (currentBatchSize <= 0) break;

                // Allocate contiguous unmanaged batched tensors: [Batch, Channels, Height, Width]
                var batchImages = new NDArray<float>(currentBatchSize, 3, TargetHeight, TargetWidth);
                var batchLabels = new NDArray<int>(currentBatchSize);

                unsafe { float* imagesPtr = batchImages.Buffer.GetUnsafePointer() + batchImages.Offset;
                int* labelsPtr = (int*)batchLabels.Buffer.GetUnsafePointer() + batchLabels.Offset;

                // Parallel batch decoding and transformation
                Parallel.For(0, currentBatchSize, i =>
                {
                    int sampleIdx = indices[b * BatchSize + i];
                    var (img, label) = Dataset.GetItem(sampleIdx);

                    labelsPtr[i] = label;

                    // Fused resize + normalize + transpose directly into the batch memory slot
                    float* targetSlot = imagesPtr + (i * singleImageElements);
                    var slotInfo = ImageInfo.Float32CHW(TargetWidth, TargetHeight, 3);
                    using var slotBuffer = new ImageBuffer(targetSlot, slotInfo, isOwner: false);

                    if (img.Width == TargetWidth && img.Height == TargetHeight && img.Layout == MemoryLayout.CHW)
                    {
                        img.CopyTo(slotBuffer);
                    }
                    else
                    {
                        // Standard ImageNet normalization parameters
                        ReadOnlySpan<float> mean = [0.485f, 0.456f, 0.406f];
                        ReadOnlySpan<float> std = [0.229f, 0.224f, 0.225f];
                        FusedTransforms.ResizeNormalizeCHW(img, TargetWidth, TargetHeight, mean, std, slotBuffer);
                    }

                    img.Dispose();
                });
                }

                await writer.WriteAsync((batchImages, batchLabels), ct).ConfigureAwait(false);
            }

            writer.Complete();
        }
        catch (Exception ex)
        {
            writer.TryComplete(ex);
        }
    }

    public async ValueTask DisposeAsync()
    {
        _cts.Cancel();
        if (_producerTask != null)
        {
            try
            {
                await _producerTask.ConfigureAwait(false);
            }
            catch
            {
                // Ignored cancellation on shutdown
            }
        }
        _cts.Dispose();
    }
}

