namespace TokenVector.Vision.Common;

/// <summary>
/// Memory arrangement of image and tensor dimensions.
/// </summary>
public enum MemoryLayout : byte
{
    /// <summary>
    /// Interleaved format [Height, Width, Channels] - standard for image storage/codecs.
    /// </summary>
    HWC = 0,

    /// <summary>
    /// Planar tensor format [Channels, Height, Width] - standard for CNN/Vision Transformer inputs.
    /// </summary>
    CHW = 1,

    /// <summary>
    /// Batched planar tensor [Batch, Channels, Height, Width] - standard for inference batching.
    /// </summary>
    NCHW = 2,

    /// <summary>
    /// Batched interleaved tensor [Batch, Height, Width, Channels] - TensorFlow / TFLite layout.
    /// </summary>
    NHWC = 3
}
