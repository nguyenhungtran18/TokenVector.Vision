using TokenVector.Vision.Common;

namespace TokenVector.Vision.Transforms;

/// <summary>
/// Core contract for high-performance zero-GC vision transformations.
/// </summary>
public interface IVisionTransform
{
    /// <summary>
    /// Applies transformation to input ImageBuffer and writes into output ImageBuffer.
    /// If output is null, a new ImageBuffer is allocated.
    /// </summary>
    ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null);

    /// <summary>
    /// Computes the output ImageInfo based on input metadata.
    /// </summary>
    ImageInfo GetOutputInfo(ImageInfo inputInfo);
}
