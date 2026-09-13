using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TokenVector.Vision.Detection;

/// <summary>
/// High-throughput, Zero-Heap Non-Maximum Suppression (NMS) Engine.
/// Filters overlapping predicted bounding boxes from YOLO, Faster R-CNN, SSD, and DETR heads.
/// </summary>
public static unsafe class NonMaximumSuppression
{
    /// <summary>
    /// Executes Standard Class-Aware Non-Maximum Suppression.
    /// </summary>
    /// <param name="boxes">Input candidate bounding boxes.</param>
    /// <param name="iouThreshold">Intersection over Union overlap threshold (e.g. 0.45 - 0.65).</param>
    /// <param name="scoreThreshold">Confidence score minimum threshold (e.g. 0.25).</param>
    /// <param name="maxOutputBoxes">Maximum number of output boxes to keep.</param>
    /// <returns>Filtered array of top non-overlapping bounding boxes.</returns>
    public static BoundingBox[] Filter(
        ReadOnlySpan<BoundingBox> boxes,
        float iouThreshold = 0.45f,
        float scoreThreshold = 0.25f,
        int maxOutputBoxes = 300)
    {
        if (boxes.IsEmpty)
            return [];

        // 1. Filter by score threshold & collect indices
        int count = boxes.Length;
        int* candidateIndices = stackalloc int[count < 2048 ? count : 2048];
        int* allocatedBuffer = null;

        if (count >= 2048)
        {
            allocatedBuffer = (int*)NativeMemory.Alloc((nuint)count, sizeof(int));
            candidateIndices = allocatedBuffer;
        }

        try
        {
            int validCount = 0;
            for (int i = 0; i < count; i++)
            {
                if (boxes[i].Score >= scoreThreshold)
                {
                    candidateIndices[validCount++] = i;
                }
            }

            if (validCount == 0)
                return [];

            // 2. Sort candidate indices descending by score (Quicksort on stack/native memory)
            SortIndicesByScore(candidateIndices, 0, validCount - 1, boxes);

            // 3. NMS suppression loop using unmanaged bitmask / flags
            byte* suppressed = stackalloc byte[validCount < 2048 ? validCount : 2048];
            byte* allocatedSuppressed = null;
            if (validCount >= 2048)
            {
                allocatedSuppressed = (byte*)NativeMemory.AllocZeroed((nuint)validCount, sizeof(byte));
                suppressed = allocatedSuppressed;
            }
            else
            {
                new Span<byte>(suppressed, validCount).Clear();
            }

            try
            {
                var resultList = new List<BoundingBox>(Math.Min(validCount, maxOutputBoxes));

                for (int i = 0; i < validCount; i++)
                {
                    if (suppressed[i] != 0)
                        continue;

                    int currentIdx = candidateIndices[i];
                    BoundingBox currentBox = boxes[currentIdx];
                    resultList.Add(currentBox);

                    if (resultList.Count >= maxOutputBoxes)
                        break;

                    // Suppress subsequent overlapping boxes of the SAME class
                    for (int j = i + 1; j < validCount; j++)
                    {
                        if (suppressed[j] != 0)
                            continue;

                        int testIdx = candidateIndices[j];
                        BoundingBox testBox = boxes[testIdx];

                        if (currentBox.ClassId == testBox.ClassId)
                        {
                            float iou = currentBox.ComputeIoU(testBox);
                            if (iou >= iouThreshold)
                            {
                                suppressed[j] = 1;
                            }
                        }
                    }
                }

                return resultList.ToArray();
            }
            finally
            {
                if (allocatedSuppressed != null)
                    NativeMemory.Free(allocatedSuppressed);
            }
        }
        finally
        {
            if (allocatedBuffer != null)
                NativeMemory.Free(allocatedBuffer);
        }
    }

    /// <summary>
    /// In-place quicksort of indices based on box score (descending).
    /// </summary>
    private static void SortIndicesByScore(int* indices, int left, int right, ReadOnlySpan<BoundingBox> boxes)
    {
        int i = left;
        int j = right;
        float pivot = boxes[indices[(left + right) / 2]].Score;

        while (i <= j)
        {
            while (boxes[indices[i]].Score > pivot) i++;
            while (boxes[indices[j]].Score < pivot) j--;

            if (i <= j)
            {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
                i++;
                j--;
            }
        }

        if (left < j) SortIndicesByScore(indices, left, j, boxes);
        if (i < right) SortIndicesByScore(indices, i, right, boxes);
    }
}
