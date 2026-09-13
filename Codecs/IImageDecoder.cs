using System;
using System.IO;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Codecs;

/// <summary>
/// Interface for high-speed image decoders and header parsers.
/// </summary>
public interface IImageDecoder
{
    /// <summary>
    /// Parses image headers in O(1) time without loading entire pixel payload.
    /// </summary>
    ImageInfo DecodeHeader(ReadOnlySpan<byte> streamData);

    /// <summary>
    /// Parses file header on disk without reading the entire file.
    /// </summary>
    ImageInfo DecodeFileHeader(string filePath);

    /// <summary>
    /// Decodes raw image data directly into an unmanaged ImageBuffer.
    /// </summary>
    ImageBuffer Decode(ReadOnlySpan<byte> streamData, ImageBuffer? targetBuffer = null);

    /// <summary>
    /// Decodes an image file on disk into an unmanaged ImageBuffer.
    /// </summary>
    ImageBuffer DecodeFile(string filePath, ImageBuffer? targetBuffer = null);
}
