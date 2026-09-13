using System.Runtime.InteropServices;
using System;
using System.Buffers.Binary;
using System.IO;
using System.Text;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Codecs;

/// <summary>
/// Ultra-fast unmanaged image decoder and header parser.
/// Supports zero-heap header parsing for JPEG, PNG, BMP, WebP, QOI, and Netpbm (PPM/PGM).
/// Direct pixel decoders for BMP, QOI, PPM, and PGM with zero GC allocation.
/// </summary>
public sealed unsafe class ImageDecoder : IImageDecoder
{
    public static ImageDecoder Instance { get; } = new();

    /// <inheritdoc/>
    public ImageInfo DecodeHeader(ReadOnlySpan<byte> data)
    {
        if (data.Length < 8)
        {
            throw new ArgumentException("Data too short to contain valid image header.", nameof(data));
        }

        // 1. PNG Header (89 50 4E 47 0D 0A 1A 0A)
        if (data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 &&
            data[4] == 0x0D && data[5] == 0x0A && data[6] == 0x1A && data[7] == 0x0A)
        {
            if (data.Length < 24) throw new InvalidDataException("Incomplete PNG IHDR chunk.");
            int width = BinaryPrimitives.ReadInt32BigEndian(data.Slice(16, 4));
            int height = BinaryPrimitives.ReadInt32BigEndian(data.Slice(20, 4));
            byte colorType = data.Length >= 26 ? data[25] : (byte)2;
            int channels = colorType switch
            {
                0 => 1, // Grayscale
                2 => 3, // Truecolor RGB
                3 => 3, // Indexed Color
                4 => 2, // Grayscale + Alpha
                6 => 4, // Truecolor RGBA
                _ => 3
            };
            var format = channels == 1 ? PixelFormat.Grayscale8 : (channels == 4 ? PixelFormat.Rgba32 : PixelFormat.Rgb24);
            return new ImageInfo(width, height, channels, format);
        }

        // 2. BMP Header ('BM' = 0x42, 0x4D)
        if (data[0] == 0x42 && data[1] == 0x4D)
        {
            if (data.Length < 26) throw new InvalidDataException("Incomplete BMP header.");
            int width = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(18, 4));
            int height = Math.Abs(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(22, 4)));
            short bpp = data.Length >= 30 ? BinaryPrimitives.ReadInt16LittleEndian(data.Slice(28, 2)) : (short)24;
            int channels = bpp switch
            {
                8 => 1,
                24 => 3,
                32 => 4,
                _ => 3
            };
            var format = channels == 1 ? PixelFormat.Grayscale8 : (channels == 4 ? PixelFormat.Rgba32 : PixelFormat.Rgb24);
            return new ImageInfo(width, height, channels, format);
        }

        // 3. JPEG Header (0xFF 0xD8)
        if (data[0] == 0xFF && data[1] == 0xD8)
        {
            int offset = 2;
            while (offset + 4 < data.Length)
            {
                if (data[offset] != 0xFF)
                {
                    offset++;
                    continue;
                }

                byte marker = data[offset + 1];
                if (marker == 0xD9 || marker == 0xDA) // EOI or SOS (start of scan)
                    break;

                int segmentLength = BinaryPrimitives.ReadUInt16BigEndian(data.Slice(offset + 2, 2));

                // SOF0 (Baseline), SOF1 (Extended), SOF2 (Progressive)
                if (marker is 0xC0 or 0xC1 or 0xC2)
                {
                    if (offset + 10 < data.Length)
                    {
                        int height = BinaryPrimitives.ReadUInt16BigEndian(data.Slice(offset + 5, 2));
                        int width = BinaryPrimitives.ReadUInt16BigEndian(data.Slice(offset + 7, 2));
                        int channels = data[offset + 9];
                        var format = channels == 1 ? PixelFormat.Grayscale8 : PixelFormat.Rgb24;
                        return new ImageInfo(width, height, channels, format);
                    }
                }

                offset += 2 + segmentLength;
            }
            throw new InvalidDataException("SOF segment not found in JPEG stream.");
        }

        // 4. QOI Header ('qoif' = 0x71, 0x6F, 0x69, 0x66)
        if (data[0] == (byte)'q' && data[1] == (byte)'o' && data[2] == (byte)'i' && data[3] == (byte)'f')
        {
            if (data.Length < 14) throw new InvalidDataException("Incomplete QOI header.");
            int width = (int)BinaryPrimitives.ReadUInt32BigEndian(data.Slice(4, 4));
            int height = (int)BinaryPrimitives.ReadUInt32BigEndian(data.Slice(8, 4));
            byte channels = data[12];
            var format = channels == 4 ? PixelFormat.Rgba32 : PixelFormat.Rgb24;
            return new ImageInfo(width, height, channels, format);
        }

        // 5. Netpbm PPM (P6) or PGM (P5)
        if (data[0] == (byte)'P' && (data[1] == (byte)'6' || data[1] == (byte)'5'))
        {
            return ParseNetpbmHeader(data);
        }

        // 6. WebP Header (RIFF....WEBP)
        if (data.Length >= 30 && data[0] == 0x52 && data[1] == 0x49 && data[2] == 0x46 && data[3] == 0x46 &&
            data[8] == 0x57 && data[9] == 0x45 && data[10] == 0x42 && data[11] == 0x50)
        {
            // VP8 (lossy)
            if (data[12] == 0x56 && data[13] == 0x50 && data[14] == 0x38 && data[15] == 0x20)
            {
                int width = (data[26] | (data[27] << 8)) & 0x3FFF;
                int height = (data[28] | (data[29] << 8)) & 0x3FFF;
                return new ImageInfo(width, height, 3, PixelFormat.Rgb24);
            }
            // VP8L (lossless)
            if (data[12] == 0x56 && data[13] == 0x50 && data[14] == 0x38 && data[15] == 0x4C)
            {
                uint b0 = data[21], b1 = data[22], b2 = data[23], b3 = data[24];
                int width = 1 + (int)(b0 | ((b1 & 0x3F) << 8));
                int height = 1 + (int)(((b1 >> 6) | (b2 << 2) | ((b3 & 0xF) << 10)));
                return new ImageInfo(width, height, 4, PixelFormat.Rgba32);
            }
        }

        throw new NotSupportedException("Unrecognized or unsupported image format magic bytes.");
    }

    /// <inheritdoc/>
    public ImageInfo DecodeFileHeader(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        Span<byte> buffer = stackalloc byte[4096];
        int bytesRead = stream.Read(buffer);
        return DecodeHeader(buffer.Slice(0, bytesRead));
    }

    /// <inheritdoc/>
    public ImageBuffer Decode(ReadOnlySpan<byte> data, ImageBuffer? targetBuffer = null)
    {
        var info = DecodeHeader(data);

        // 1. BMP Fast Unmanaged Decoder
        if (data[0] == 0x42 && data[1] == 0x4D)
        {
            return DecodeBmp(data, info, targetBuffer);
        }

        // 2. QOI Fast Unmanaged Decoder
        if (data[0] == (byte)'q' && data[1] == (byte)'o' && data[2] == (byte)'i' && data[3] == (byte)'f')
        {
            return DecodeQoi(data, info, targetBuffer);
        }

        // 3. Netpbm (PPM/PGM) Fast Decoder
        if (data[0] == (byte)'P' && (data[1] == (byte)'6' || data[1] == (byte)'5'))
        {
            return DecodeNetpbm(data, info, targetBuffer);
        }

        // 4. Default: Allocate buffer based on info
        var buffer = targetBuffer ?? new ImageBuffer(info);
        return buffer;
    }

    /// <inheritdoc/>
    public ImageBuffer DecodeFile(string filePath, ImageBuffer? targetBuffer = null)
    {
        byte[] fileBytes = File.ReadAllBytes(filePath);
        return Decode(fileBytes, targetBuffer);
    }

    #region Internal Format Parsers

    private static ImageInfo ParseNetpbmHeader(ReadOnlySpan<byte> data)
    {
        int channels = data[1] == (byte)'6' ? 3 : 1;
        int i = 2;
        // Skip whitespace and comments
        while (i < data.Length && (data[i] == ' ' || data[i] == '\t' || data[i] == '\r' || data[i] == '\n')) i++;
        while (i < data.Length && data[i] == '#')
        {
            while (i < data.Length && data[i] != '\n') i++;
            while (i < data.Length && (data[i] == ' ' || data[i] == '\t' || data[i] == '\r' || data[i] == '\n')) i++;
        }

        int width = 0;
        while (i < data.Length && char.IsAsciiDigit((char)data[i]))
        {
            width = width * 10 + (data[i] - '0');
            i++;
        }
        while (i < data.Length && (data[i] == ' ' || data[i] == '\t' || data[i] == '\r' || data[i] == '\n')) i++;

        int height = 0;
        while (i < data.Length && char.IsAsciiDigit((char)data[i]))
        {
            height = height * 10 + (data[i] - '0');
            i++;
        }

        var format = channels == 1 ? PixelFormat.Grayscale8 : PixelFormat.Rgb24;
        return new ImageInfo(width, height, channels, format);
    }

    private static ImageBuffer DecodeBmp(ReadOnlySpan<byte> data, ImageInfo info, ImageBuffer? targetBuffer)
    {
        int pixelDataOffset = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(10, 4));
        int height = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(22, 4));
        bool isTopDown = height < 0;
        int absHeight = Math.Abs(height);
        int bpp = BinaryPrimitives.ReadInt16LittleEndian(data.Slice(28, 2));

        var buffer = targetBuffer ?? new ImageBuffer(info);
        int rowBytes = (info.Width * bpp + 31) / 32 * 4; // 4-byte row alignment in BMP

        byte* dstBase = buffer.BytePointer;
        fixed (byte* srcBase = data)
        {
            byte* srcPixels = srcBase + pixelDataOffset;

            for (int y = 0; y < absHeight; y++)
            {
                int srcY = isTopDown ? y : (absHeight - 1 - y);
                byte* srcRow = srcPixels + (srcY * rowBytes);
                byte* dstRow = dstBase + (y * buffer.Stride);

                if (bpp == 24 && (buffer.Format == PixelFormat.Rgb24 || buffer.Channels == 3))
                {
                    // Convert BGR -> RGB
                    for (int x = 0; x < info.Width; x++)
                    {
                        byte b = srcRow[x * 3];
                        byte g = srcRow[x * 3 + 1];
                        byte r = srcRow[x * 3 + 2];
                        dstRow[x * 3] = r;
                        dstRow[x * 3 + 1] = g;
                        dstRow[x * 3 + 2] = b;
                    }
                }
                else
                {
                    int copyBytes = Math.Min(info.Width * info.Channels, rowBytes);
                    Buffer.MemoryCopy(srcRow, dstRow, buffer.Stride, copyBytes);
                }
            }
        }
        return buffer;
    }

    private static ImageBuffer DecodeQoi(ReadOnlySpan<byte> data, ImageInfo info, ImageBuffer? targetBuffer)
    {
        var buffer = targetBuffer ?? new ImageBuffer(info);
        int channels = info.Channels;
        int totalPixels = info.Width * info.Height;

        // QOI running array of 64 recently seen pixels
        byte* index = stackalloc byte[64 * 4];
        NativeMemory.Clear(index, 64 * 4);

        byte r = 0, g = 0, b = 0, a = 255;
        int pixelPos = 0;
        int p = 14; // Header size is 14 bytes
        byte* dst = buffer.BytePointer;

        while (pixelPos < totalPixels && p < data.Length)
        {
            byte b1 = data[p++];

            if (b1 == 0xFE) // QOI_OP_RGB
            {
                r = data[p++];
                g = data[p++];
                b = data[p++];
            }
            else if (b1 == 0xFF) // QOI_OP_RGBA
            {
                r = data[p++];
                g = data[p++];
                b = data[p++];
                a = data[p++];
            }
            else if ((b1 & 0xC0) == 0x00) // QOI_OP_INDEX
            {
                int idx = b1 * 4;
                r = index[idx];
                g = index[idx + 1];
                b = index[idx + 2];
                a = index[idx + 3];
            }
            else if ((b1 & 0xC0) == 0x40) // QOI_OP_DIFF
            {
                r += (byte)(((b1 >> 4) & 0x03) - 2);
                g += (byte)(((b1 >> 2) & 0x03) - 2);
                b += (byte)((b1 & 0x03) - 2);
            }
            else if ((b1 & 0xC0) == 0x80) // QOI_OP_LUMA
            {
                byte b2 = data[p++];
                int vg = (b1 & 0x3F) - 32;
                r += (byte)(vg - 8 + ((b2 >> 4) & 0x0F));
                g += (byte)vg;
                b += (byte)(vg - 8 + (b2 & 0x0F));
            }
            else if ((b1 & 0xC0) == 0xC0) // QOI_OP_RUN
            {
                int run = (b1 & 0x3F);
                for (int i = 0; i <= run && pixelPos < totalPixels; i++)
                {
                    int outIdx = pixelPos * channels;
                    dst[outIdx] = r;
                    dst[outIdx + 1] = g;
                    dst[outIdx + 2] = b;
                    if (channels == 4) dst[outIdx + 3] = a;
                    pixelPos++;
                }
                continue;
            }

            int hash = ((r * 3 + g * 5 + b * 7 + a * 11) % 64) * 4;
            index[hash] = r;
            index[hash + 1] = g;
            index[hash + 2] = b;
            index[hash + 3] = a;

            int dstOffset = pixelPos * channels;
            dst[dstOffset] = r;
            dst[dstOffset + 1] = g;
            dst[dstOffset + 2] = b;
            if (channels == 4) dst[dstOffset + 3] = a;
            pixelPos++;
        }

        return buffer;
    }

    private static ImageBuffer DecodeNetpbm(ReadOnlySpan<byte> data, ImageInfo info, ImageBuffer? targetBuffer)
    {
        var buffer = targetBuffer ?? new ImageBuffer(info);
        int i = 2;
        // Skip header lines to reach payload
        int lines = 0;
        while (i < data.Length && lines < 3)
        {
            if (data[i] == '\n') lines++;
            i++;
        }

        int payloadBytes = info.Width * info.Height * info.Channels;
        int available = data.Length - i;
        int copyBytes = Math.Min(payloadBytes, available);

        fixed (byte* src = data.Slice(i))
        {
            Buffer.MemoryCopy(src, buffer.BytePointer, buffer.ByteLength, copyBytes);
        }
        return buffer;
    }

    #endregion
}


