using System;
using System.IO;
using System.Threading.Tasks;
using TokenVector.Vision.Common;
using TokenVector.Vision.Datasets;
using TokenVector.Vision.Transforms;
using Xunit;

namespace TokenVector.Vision.Tests;

public class DataLoaderTests
{
    [Fact]
    public async Task VisionDataLoader_PrefetchesBatchesAsync()
    {
        // Setup mock dataset folder structure
        string tempDir = Path.Combine(Path.GetTempPath(), "tkv_vision_test_" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(Path.Combine(tempDir, "cat"));
            Directory.CreateDirectory(Path.Combine(tempDir, "dog"));

            // Create mock BMP images
            using (var catImg = ImageBuffer.CreateRgb(32, 32))
            {
                catImg.AsByteSpan().Fill(100);
                Codecs.ImageEncoder.Instance.EncodeToFile(catImg, Path.Combine(tempDir, "cat", "cat1.bmp"));
                Codecs.ImageEncoder.Instance.EncodeToFile(catImg, Path.Combine(tempDir, "cat", "cat2.bmp"));
            }

            using (var dogImg = ImageBuffer.CreateRgb(32, 32))
            {
                dogImg.AsByteSpan().Fill(200);
                Codecs.ImageEncoder.Instance.EncodeToFile(dogImg, Path.Combine(tempDir, "dog", "dog1.bmp"));
                Codecs.ImageEncoder.Instance.EncodeToFile(dogImg, Path.Combine(tempDir, "dog", "dog2.bmp"));
            }

            var dataset = new ImageFolderDataset(tempDir);
            Assert.Equal(4, dataset.Count);
            Assert.Equal(2, dataset.Classes.Count);

            await using var loader = new VisionDataLoader(dataset, batchSize: 2, shuffle: false, targetWidth: 16, targetHeight: 16);

            int batchCount = 0;
            await foreach (var (images, labels) in loader.GetBatchesAsync())
            {
                Assert.Equal(4, images.Rank);
                Assert.Equal(2, images.Shape[0]); // Batch size 2
                Assert.Equal(3, images.Shape[1]); // 3 Channels
                Assert.Equal(16, images.Shape[2]); // Height
                Assert.Equal(16, images.Shape[3]); // Width

                batchCount++;
                images.Dispose();
                labels.Dispose();
            }

            Assert.Equal(2, batchCount); // 4 items / batch 2 = 2 batches
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, true);
            }
        }
    }
}
