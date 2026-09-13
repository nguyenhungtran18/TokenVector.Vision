using TokenVector.Vision.Common;

namespace TokenVector.Vision.Filters;

/// <summary>
/// Base contract for high-performance 2D spatial image filters.
/// </summary>
public interface IVisionFilter
{
    /// <summary>
    /// Applies spatial filtering to input ImageBuffer.
    /// </summary>
    ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null);
}
