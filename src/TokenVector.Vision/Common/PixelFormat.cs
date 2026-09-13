namespace TokenVector.Vision.Common;

/// <summary>
/// Supported pixel formats for vision processing and deep learning pipelines.
/// </summary>
public enum PixelFormat : byte
{
    /// <summary>
    /// Single channel 8-bit unsigned integer grayscale [0, 255].
    /// </summary>
    Grayscale8 = 1,

    /// <summary>
    /// 3-channel 8-bit RGB interleaved (Red, Green, Blue).
    /// </summary>
    Rgb24 = 3,

    /// <summary>
    /// 3-channel 8-bit BGR interleaved (Blue, Green, Red) - OpenCV style.
    /// </summary>
    Bgr24 = 4,

    /// <summary>
    /// 4-channel 8-bit RGBA interleaved (Red, Green, Blue, Alpha).
    /// </summary>
    Rgba32 = 5,

    /// <summary>
    /// 4-channel 8-bit BGRA interleaved (Blue, Green, Red, Alpha).
    /// </summary>
    Bgra32 = 6,

    /// <summary>
    /// Single channel 32-bit floating point [0.0, 1.0] or arbitrary feature map.
    /// </summary>
    Float32Grayscale = 10,

    /// <summary>
    /// 3-channel 32-bit floating point RGB interleaved (HWC).
    /// </summary>
    Float32Rgb = 11,

    /// <summary>
    /// 3-channel 32-bit floating point planar tensor (CHW).
    /// </summary>
    Float32Planar = 12,

    /// <summary>
    /// 4-channel 32-bit floating point RGBA.
    /// </summary>
    Float32Rgba = 13
}
