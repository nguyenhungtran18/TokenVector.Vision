namespace TokenVector.Vision.Common;

/// <summary>
/// Image sampling and spatial interpolation algorithms.
/// </summary>
public enum InterpolationMode : byte
{
    /// <summary>
    /// Nearest-neighbor interpolation (fastest, box filter).
    /// </summary>
    Nearest = 0,

    /// <summary>
    /// Bilinear interpolation using 2x2 neighborhood (default for standard vision tasks).
    /// </summary>
    Bilinear = 1,

    /// <summary>
    /// Bicubic interpolation using 4x4 neighborhood (Catmull-Rom or cubic spline).
    /// </summary>
    Bicubic = 2,

    /// <summary>
    /// Lanczos-3 interpolation using 6x6 sinc window (highest quality, sharpest details).
    /// </summary>
    Lanczos = 3
}
