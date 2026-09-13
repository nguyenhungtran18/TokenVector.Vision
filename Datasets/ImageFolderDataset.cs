using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TokenVector.Vision.Codecs;
using TokenVector.Vision.Common;
using TokenVector.Vision.Transforms;

namespace TokenVector.Vision.Datasets;

/// <summary>
/// High-performance dataset reader for directory structure: root_dir/class_name/xxx.ext
/// Supports lazy parallel decoding, automatic class index mapping, and transform injection.
/// </summary>
public sealed class ImageFolderDataset
{
    private static readonly HashSet<string> SupportedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".qoi", ".ppm", ".pgm", ".webp"
    };

    public string RootDirectory { get; }
    public IReadOnlyList<string> Classes { get; }
    public IReadOnlyDictionary<string, int> ClassToIdx { get; }
    public IReadOnlyList<(string FilePath, int ClassIndex)> Samples { get; }
    public ComposePipeline? Transform { get; set; }

    public int Count => Samples.Count;

    public ImageFolderDataset(string rootDirectory, ComposePipeline? transform = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(rootDirectory);
        if (!Directory.Exists(rootDirectory))
            throw new DirectoryNotFoundException($"Dataset directory not found: {rootDirectory}");

        RootDirectory = rootDirectory;
        Transform = transform;

        // 1. Discover class folders
        var classDirs = Directory.GetDirectories(rootDirectory)
                                 .Select(Path.GetFileName)
                                 .Where(name => !string.IsNullOrEmpty(name))
                                 .OrderBy(name => name, StringComparer.Ordinal)
                                 .ToArray();

        Classes = classDirs!;
        var classToIdxMap = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < classDirs.Length; i++)
        {
            classToIdxMap[classDirs[i]!] = i;
        }
        ClassToIdx = classToIdxMap;

        // 2. Discover all image files
        var samplesList = new List<(string FilePath, int ClassIndex)>();
        foreach (var (className, classIdx) in classToIdxMap)
        {
            string classPath = Path.Combine(rootDirectory, className);
            var files = Directory.EnumerateFiles(classPath, "*.*", SearchOption.AllDirectories)
                                 .Where(f => SupportedExtensions.Contains(Path.GetExtension(f)))
                                 .OrderBy(f => f, StringComparer.Ordinal);

            foreach (var file in files)
            {
                samplesList.Add((file, classIdx));
            }
        }

        Samples = samplesList;
    }

    /// <summary>
    /// Loads and transforms a single sample by index.
    /// </summary>
    public (ImageBuffer Image, int Label) GetItem(int index)
    {
        if ((uint)index >= (uint)Samples.Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        var (filePath, label) = Samples[index];
        var image = ImageDecoder.Instance.DecodeFile(filePath);

        if (Transform != null)
        {
            var transformed = Transform.Execute(image);
            return (transformed, label);
        }

        return (image, label);
    }
}
