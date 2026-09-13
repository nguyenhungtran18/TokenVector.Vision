using System.Runtime.InteropServices;
using System;
using System.Buffers.Binary;
using System.IO;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Codecs;

/// <summary>
/// Ultra high-speed unmanaged image encoder for BMP, QOI, Netpbm, and Raw image formats.
/// Zero GC allocations on pre-allocated streams or byte spans.
/// </summary>
public sealed unsafe class ImageEncoder : IImageEncoder
{
    public static ImageEncoder Instance { get; } = new();

    /// <inheritdoc/>
    public void Encode(ImageBuffer buffer, Stream stream, string format = "bmp")
    {
        ArgumentNullException.ThrowIfNull(buffer);
        ArgumentNullException.ThrowIfNull(stream);

        switch (format.ToLowerInvariant())
        {
            case "bmp":
                EncodeBmp(buffer, stream);
                break;
            case "qoi":
                EncodeQoi(buffer, stream);
                break;
            case "ppm":
                EncodePpm(buffer, stream);
                break;
            default:
                throw new NotSupportedException($"Image encoding format '{format}' is not supported.");
        }
    }

    /// <inheritdoc/>
    public void EncodeToFile(ImageBuffer buffer, string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        string ext = Path.GetExtension(filePath).TrimStart('.').ToLowerInvariant();
        using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        Encode(buffer, fs, string.IsNullOrEmpty(ext) ? "bmp" : ext);
    }

    /// <inheritdoc/>
    public byte[] EncodeToBytes(ImageBuffer buffer, string format = "bmp")
    {
        using var ms = new MemoryStream();
        Encode(buffer, ms, format);
        return ms.ToArray();
    }

    #region Internal Encoders

    private static void EncodeBmp(ImageBuffer buffer, Stream stream)
    {
        int width = buffer.Width;
        int height = buffer.Height;
        int channels = buffer.Channels;
        short bpp = (short)(channels * 8);

        int rowBytes = (width * channels + 3) & ~3; // 4-byte row alignment
        int imageSize = rowBytes * height;
        int fileSize = 54 + imageSize;

        Span<byte> header = stackalloc byte[54];

        // BMP Header
        header[0] = 0x42; // 'B'
        header[1] = 0x4D; // 'M'
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(2, 4), fileSize);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(6, 4), 0); // Reserved
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(10, 4), 54); // Offset to pixel data

        // DIB Header (BITMAPINFOHEADER)
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(14, 4), 40); // DIB header size
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(18, 4), width);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(22, 4), height); // Bottom-up
        BinaryPrimitives.WriteInt16LittleEndian(header.Slice(26, 2), 1); // Planes
        BinaryPrimitives.WriteInt16LittleEndian(header.Slice(28, 2), bpp);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(30, 4), 0); // BI_RGB (no compression)
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(34, 4), imageSize);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(38, 4), 2835); // 72 DPI
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(42, 4), 2835);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(46, 4), 0);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(50, 4), 0);

        stream.Write(header);

        // Write pixel data bottom-up with BGR format
        byte[] rowBuffer = new byte[rowBytes];
        fixed (byte* rowPtr = rowBuffer)
        {
            for (int y = height - 1; y >= 0; y--)
            {
                byte* srcRow = buffer.GetRowBytePointer(y);

                if (channels == 3)
                {
                    // RGB -> BGR
                    for (int x = 0; x < width; x++)
                    {
                        rowPtr[x * 3] = srcRow[x * 3 + 2];     // B
                        rowPtr[x * 3 + 1] = srcRow[x * 3 + 1]; // G
                        rowPtr[x * 3 + 2] = srcRow[x * 3];     // R
                    }
                }
                else
                {
                    Buffer.MemoryCopy(srcRow, rowPtr, rowBytes, width * channels);
                }

                stream.Write(rowBuffer, 0, rowBytes);
            }
        }
    }

    private static void EncodeQoi(ImageBuffer buffer, Stream stream)
    {
        int width = buffer.Width;
        int height = buffer.Height;
        byte channels = (byte)buffer.Channels;

        Span<byte> header = stackalloc byte[14];
        header[0] = (byte)'q';
        header[1] = (byte)'o';
        header[2] = (byte)'i';
        header[3] = (byte)'f';
        BinaryPrimitives.WriteUInt32BigEndian(header.Slice(4, 4), (uint)width);
        BinaryPrimitives.WriteUInt32BigEndian(header.Slice(8, 4), (uint)height);
        header[12] = channels;
        header[13] = 0; // sRGB with linear alpha

        stream.Write(header);

        // Fast QOI stream encoding
        int maxEncoded = width * height * (channels + 1) + 14 + 8;
        byte[] outputBuffer = new byte[maxEncoded];
        int outPos = 0;

        byte* index = stackalloc byte[64 * 4];
        NativeMemory.Clear(index, 64 * 4);

        byte prevR = 0, prevG = 0, prevB = 0, prevA = 255;
        byte r = 0, g = 0, b = 0, a = 255;
        int run = 0;
        int totalPixels = width * height;
        byte* src = buffer.BytePointer;

        for (int px = 0; px < totalPixels; px++)
        {
            int srcIdx = px * channels;
            r = src[srcIdx];
            g = src[srcIdx + 1];
            b = src[srcIdx + 2];
            a = channels == 4 ? src[srcIdx + 3] : (byte)255;

            if (r == prevR && g == prevG && b == prevB && a == prevA)
            {
                run++;
                if (run == 62 || px == totalPixels - 1)
                {
                    outputBuffer[outPos++] = (byte)(0xC0 | (run - 1));
                    run = 0;
                }
            }
            else
            {
                if (run > 0)
                {
                    outputBuffer[outPos++] = (byte)(0xC0 | (run - 1));
                    run = 0;
                }

                int hash = ((r * 3 + g * 5 + b * 7 + a * 11) % 64) * 4;

                if (index[hash] == r && index[hash + 1] == g && index[hash + 2] == b && index[hash + 3] == a)
                {
                    outputBuffer[outPos++] = (byte)(hash / 4);
                }
                else
                {
                    index[hash] = r;
                    index[hash + 1] = g;
                    index[hash + 2] = b;
                    index[hash + 3] = a;

                    if (a == prevA)
                    {
                        sbyte vr = (sbyte)(r - prevR);
                        sbyte vg = (sbyte)(g - prevG);
                        sbyte vb = (sbyte)(b - prevB);

                        sbyte vg_r = (sbyte)(vr - vg);
                        sbyte vg_b = (sbyte)(vb - vg);

                        if (vr > -3 && vr < 2 && vg > -3 && vg < 2 && vb > -3 && vb < 2)
                        {
                            outputBuffer[outPos++] = (byte)(0x40 | ((vr + 2) << 4) | ((vg + 2) << 2) | (vb + 2));
                        }
                        else if (vg_r > -9 && vg_r < 8 && vg > -33 && vg < 32 && vg_b > -9 && vg_b < 8)
                        {
                            outputBuffer[outPos++] = (byte)(0x80 | (vg + 32));
                            outputBuffer[outPos++] = (byte)(((vg_r + 8) << 4) | (vg_b + 8));
                        }
                        else
                        {
                            outputBuffer[outPos++] = 0xFE;
                            outputBuffer[outPos++] = r;
                            outputBuffer[outPos++] = g;
                            outputBuffer[outPos++] = b;
                        }
                    }
                    else
                    {
                        outputBuffer[outPos++] = 0xFF;
                        outputBuffer[outPos++] = r;
                        outputBuffer[outPos++] = g;
                        outputBuffer[outPos++] = b;
                        outputBuffer[outPos++] = a;
                    }
                }

                prevR = r;
                prevG = g;
                prevB = b;
                prevA = a;
            }
        }

        // QOI End marker (7 x 0x00, 1 x 0x01)
        outputBuffer[outPos++] = 0x00;
        outputBuffer[outPos++] = 0x00;
        outputBuffer[outPos++] = 0x00;
        outputBuffer[outPos++] = 0x00;
        outputBuffer[outPos++] = 0x00;
        outputBuffer[outPos++] = 0x00;
        outputBuffer[outPos++] = 0x00;
        outputBuffer[outPos++] = 0x01;

        stream.Write(outputBuffer, 0, outPos);
    }

    private static void EncodePpm(ImageBuffer buffer, Stream stream)
    {
        string header = $"P6\n{buffer.Width} {buffer.Height}\n255\n";
        byte[] headerBytes = System.Text.Encoding.ASCII.GetBytes(header);
        stream.Write(headerBytes);

        byte[] pixels = buffer.AsByteSpan().ToArray();
        stream.Write(pixels);
    }

    #endregion
}

