using System.IO;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Codecs;

/// <summary>
/// Interface for high-speed image encoders.
/// </summary>
public interface IImageEncoder
{
    /// <summary>
    /// Encodes an ImageBuffer into a Stream.
    /// </summary>
    void Encode(ImageBuffer buffer, Stream stream, string format = "bmp");

    /// <summary>
    /// Encodes an ImageBuffer directly into a file on disk.
    /// </summary>
    void EncodeToFile(ImageBuffer buffer, string filePath);

    /// <summary>
    /// Encodes an ImageBuffer into a newly allocated byte array.
    /// </summary>
    byte[] EncodeToBytes(ImageBuffer buffer, string format = "bmp");
}
