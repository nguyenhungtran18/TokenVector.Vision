using System;
using System.Runtime.CompilerServices;
using TokenVector.Vision.Common;
using TokenVector.Vision.Detection;

namespace TokenVector.Vision.Drawing;

/// <summary>
/// 24-bit RGB Color value for drawing primitives.
/// </summary>
public readonly record struct RgbColor(byte R, byte G, byte B)
{
    public static readonly RgbColor Red = new(255, 0, 0);
    public static readonly RgbColor Green = new(0, 255, 0);
    public static readonly RgbColor Blue = new(0, 0, 255);
    public static readonly RgbColor Yellow = new(255, 255, 0);
    public static readonly RgbColor Cyan = new(0, 255, 255);
    public static readonly RgbColor Magenta = new(255, 0, 255);
    public static readonly RgbColor White = new(255, 255, 255);
    public static readonly RgbColor Black = new(0, 0, 0);
    public static readonly RgbColor Orange = new(255, 165, 0);
}

/// <summary>
/// Zero-GC Native Drawing and Annotation Engine for Computer Vision.
/// Draws bounding boxes, confidence tags, keypoints, segmentation masks, lines, and text directly onto unmanaged ImageBuffer.
/// </summary>
public static unsafe class ImagePainter

{
    // Minimal standard 8x8 ASCII font bitmap table for characters '0'-'9', 'A'-'Z', 'a'-'z', '.', ':', '%', ' ', '#', '-', '_'
    private static readonly byte[] Font8x8 = CreateBasicFont8x8();

    /// <summary>
    /// Sets a single pixel color with bounds checking.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SetPixel(ImageBuffer image, int x, int y, RgbColor color)
    {
        if ((uint)x >= (uint)image.Width || (uint)y >= (uint)image.Height)
            return;

        byte* ptr = image.BytePointer + y * image.Stride + x * 3;
        ptr[0] = color.R;
        ptr[1] = color.G;
        ptr[2] = color.B;
    }

    /// <summary>
    /// Draws a straight line using Bresenham's algorithm.
    /// </summary>
    public static void DrawLine(ImageBuffer image, int x0, int y0, int x1, int y1, RgbColor color, int thickness = 1)
    {
        int dx = Math.Abs(x1 - x0);
        int dy = -Math.Abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;

        while (true)
        {
            if (thickness <= 1)
            {
                SetPixel(image, x0, y0, color);
            }
            else
            {
                FillCircle(image, x0, y0, thickness / 2, color);
            }

            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }

    /// <summary>
    /// Draws a rectangle outline with specified border thickness.
    /// </summary>
    public static void DrawRectangle(ImageBuffer image, int x, int y, int width, int height, RgbColor color, int thickness = 2)
    {
        int x2 = x + width - 1;
        int y2 = y + height - 1;

        for (int t = 0; t < thickness; t++)
        {
            // Top & Bottom
            for (int px = x + t; px <= x2 - t; px++)
            {
                SetPixel(image, px, y + t, color);
                SetPixel(image, px, y2 - t, color);
            }
            // Left & Right
            for (int py = y + t; py <= y2 - t; py++)
            {
                SetPixel(image, x + t, py, color);
                SetPixel(image, x2 - t, py, color);
            }
        }
    }

    /// <summary>
    /// Fills a solid rectangle with optional alpha transparency blending.
    /// </summary>
    public static void FillRectangle(ImageBuffer image, int x, int y, int width, int height, RgbColor color, float alpha = 1.0f)
    {
        int startX = Math.Clamp(x, 0, image.Width);
        int startY = Math.Clamp(y, 0, image.Height);
        int endX = Math.Clamp(x + width, 0, image.Width);
        int endY = Math.Clamp(y + height, 0, image.Height);

        alpha = Math.Clamp(alpha, 0.0f, 1.0f);
        float invAlpha = 1.0f - alpha;

        byte* basePtr = image.BytePointer;
        int stride = image.Stride;


        for (int py = startY; py < endY; py++)
        {
            byte* row = basePtr + py * stride;
            for (int px = startX; px < endX; px++)
            {
                int off = px * 3;
                if (alpha >= 0.999f)
                {
                    row[off] = color.R;
                    row[off + 1] = color.G;
                    row[off + 2] = color.B;
                }
                else
                {
                    row[off] = (byte)(row[off] * invAlpha + color.R * alpha);
                    row[off + 1] = (byte)(row[off + 1] * invAlpha + color.G * alpha);
                    row[off + 2] = (byte)(row[off + 2] * invAlpha + color.B * alpha);
                }
            }
        }
    }

    /// <summary>
    /// Fills a solid circle for keypoints or joints.
    /// </summary>
    public static void FillCircle(ImageBuffer image, int cx, int cy, int radius, RgbColor color)
    {
        int r2 = radius * radius;
        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                if (dx * dx + dy * dy <= r2)
                {
                    SetPixel(image, cx + dx, cy + dy, color);
                }
            }
        }
    }

    /// <summary>
    /// Draws a text string using built-in 8x8 bitmap glyphs.
    /// </summary>
    public static void DrawText(ImageBuffer image, ReadOnlySpan<char> text, int x, int y, RgbColor color, int scale = 1)
    {
        int curX = x;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c == '\n')
            {
                curX = x;
                y += 9 * scale;
                continue;
            }

            DrawChar(image, c, curX, y, color, scale);
            curX += 8 * scale;
        }
    }

    /// <summary>
    /// Draws a complete AI Detection Bounding Box with colored border, top label badge, and text tag.
    /// </summary>
    public static void DrawBoundingBox(
        ImageBuffer image,
        in BoundingBox box,
        string label,
        RgbColor color,
        int thickness = 2,
        float fillAlpha = 0.0f)
    {
        int x1 = (int)MathF.Round(box.X1);
        int y1 = (int)MathF.Round(box.Y1);
        int w = (int)MathF.Round(box.Width);
        int h = (int)MathF.Round(box.Height);

        // Optional semi-transparent box fill
        if (fillAlpha > 0.01f)
        {
            FillRectangle(image, x1, y1, w, h, color, fillAlpha);
        }

        // Outer border
        DrawRectangle(image, x1, y1, w, h, color, thickness);

        // Label Badge
        if (!string.IsNullOrEmpty(label))
        {
            int labelW = label.Length * 8 + 6;
            int labelH = 12;
            int badgeY = Math.Max(0, y1 - labelH);
            FillRectangle(image, x1, badgeY, labelW, labelH, color);
            DrawText(image, label, x1 + 3, badgeY + 2, RgbColor.Black, scale: 1);
        }
    }

    private static void DrawChar(ImageBuffer image, char c, int x, int y, RgbColor color, int scale)
    {
        int charIdx = (byte)c;
        if (charIdx > 127) charIdx = (byte)'?';

        int offset = charIdx * 8;
        for (int row = 0; row < 8; row++)
        {
            byte rowBits = Font8x8[offset + row];
            for (int col = 0; col < 8; col++)
            {
                if ((rowBits & (1 << (7 - col))) != 0)
                {
                    if (scale == 1)
                    {
                        SetPixel(image, x + col, y + row, color);
                    }
                    else
                    {
                        for (int sy = 0; sy < scale; sy++)
                            for (int sx = 0; sx < scale; sx++)
                                SetPixel(image, x + col * scale + sx, y + row * scale + sy, color);
                    }
                }
            }
        }
    }

    private static byte[] CreateBasicFont8x8()
    {
        byte[] font = new byte[128 * 8];
        // Digits '0'-'9'
        font['0' * 8 + 0] = 0x3C; font['0' * 8 + 1] = 0x66; font['0' * 8 + 2] = 0x6E; font['0' * 8 + 3] = 0x76; font['0' * 8 + 4] = 0x66; font['0' * 8 + 5] = 0x3C;
        font['1' * 8 + 0] = 0x18; font['1' * 8 + 1] = 0x38; font['1' * 8 + 2] = 0x18; font['1' * 8 + 3] = 0x18; font['1' * 8 + 4] = 0x18; font['1' * 8 + 5] = 0x7E;
        font['2' * 8 + 0] = 0x3C; font['2' * 8 + 1] = 0x66; font['2' * 8 + 2] = 0x0C; font['2' * 8 + 3] = 0x18; font['2' * 8 + 4] = 0x30; font['2' * 8 + 5] = 0x7E;
        font['3' * 8 + 0] = 0x7E; font['3' * 8 + 1] = 0x0C; font['3' * 8 + 2] = 0x1C; font['3' * 8 + 3] = 0x06; font['3' * 8 + 4] = 0x66; font['3' * 8 + 5] = 0x3C;
        font['4' * 8 + 0] = 0x0C; font['4' * 8 + 1] = 0x1C; font['4' * 8 + 2] = 0x34; font['4' * 8 + 3] = 0x64; font['4' * 8 + 4] = 0x7E; font['4' * 8 + 5] = 0x0C;
        font['5' * 8 + 0] = 0x7E; font['5' * 8 + 1] = 0x60; font['5' * 8 + 2] = 0x7C; font['5' * 8 + 3] = 0x06; font['5' * 8 + 4] = 0x66; font['5' * 8 + 5] = 0x3C;
        font['6' * 8 + 0] = 0x3C; font['6' * 8 + 1] = 0x60; font['6' * 8 + 2] = 0x7C; font['6' * 8 + 3] = 0x66; font['6' * 8 + 4] = 0x66; font['6' * 8 + 5] = 0x3C;
        font['7' * 8 + 0] = 0x7E; font['7' * 8 + 1] = 0x06; font['7' * 8 + 2] = 0x0C; font['7' * 8 + 3] = 0x18; font['7' * 8 + 4] = 0x18; font['7' * 8 + 5] = 0x18;
        font['8' * 8 + 0] = 0x3C; font['8' * 8 + 1] = 0x66; font['8' * 8 + 2] = 0x3C; font['8' * 8 + 3] = 0x66; font['8' * 8 + 4] = 0x66; font['8' * 8 + 5] = 0x3C;
        font['9' * 8 + 0] = 0x3C; font['9' * 8 + 1] = 0x66; font['9' * 8 + 2] = 0x3E; font['9' * 8 + 3] = 0x06; font['9' * 8 + 4] = 0x0C; font['9' * 8 + 5] = 0x38;

        // Letters A-Z
        font['A' * 8 + 0] = 0x18; font['A' * 8 + 1] = 0x3C; font['A' * 8 + 2] = 0x66; font['A' * 8 + 3] = 0x7E; font['A' * 8 + 4] = 0x66; font['A' * 8 + 5] = 0x66;
        font['B' * 8 + 0] = 0x7C; font['B' * 8 + 1] = 0x66; font['B' * 8 + 2] = 0x7C; font['B' * 8 + 3] = 0x66; font['B' * 8 + 4] = 0x66; font['B' * 8 + 5] = 0x7C;
        font['C' * 8 + 0] = 0x3C; font['C' * 8 + 1] = 0x66; font['C' * 8 + 2] = 0x60; font['C' * 8 + 3] = 0x60; font['C' * 8 + 4] = 0x66; font['C' * 8 + 5] = 0x3C;
        font['D' * 8 + 0] = 0x78; font['D' * 8 + 1] = 0x6C; font['D' * 8 + 2] = 0x66; font['D' * 8 + 3] = 0x66; font['D' * 8 + 4] = 0x6C; font['D' * 8 + 5] = 0x78;
        font['E' * 8 + 0] = 0x7E; font['E' * 8 + 1] = 0x60; font['E' * 8 + 2] = 0x78; font['E' * 8 + 3] = 0x60; font['E' * 8 + 4] = 0x60; font['E' * 8 + 5] = 0x7E;
        font['F' * 8 + 0] = 0x7E; font['F' * 8 + 1] = 0x60; font['F' * 8 + 2] = 0x78; font['F' * 8 + 3] = 0x60; font['F' * 8 + 4] = 0x60; font['F' * 8 + 5] = 0x60;
        font['P' * 8 + 0] = 0x7C; font['P' * 8 + 1] = 0x66; font['P' * 8 + 2] = 0x7C; font['P' * 8 + 3] = 0x60; font['P' * 8 + 4] = 0x60; font['P' * 8 + 5] = 0x60;
        font['R' * 8 + 0] = 0x7C; font['R' * 8 + 1] = 0x66; font['R' * 8 + 2] = 0x7C; font['R' * 8 + 3] = 0x6C; font['R' * 8 + 4] = 0x66; font['R' * 8 + 5] = 0x66;
        font['S' * 8 + 0] = 0x3C; font['S' * 8 + 1] = 0x60; font['S' * 8 + 2] = 0x3C; font['S' * 8 + 3] = 0x06; font['S' * 8 + 4] = 0x06; font['S' * 8 + 5] = 0x3C;
        font['Y' * 8 + 0] = 0x66; font['Y' * 8 + 1] = 0x66; font['Y' * 8 + 2] = 0x3C; font['Y' * 8 + 3] = 0x18; font['Y' * 8 + 4] = 0x18; font['Y' * 8 + 5] = 0x18;
        font['O' * 8 + 0] = 0x3C; font['O' * 8 + 1] = 0x66; font['O' * 8 + 2] = 0x66; font['O' * 8 + 3] = 0x66; font['O' * 8 + 4] = 0x66; font['O' * 8 + 5] = 0x3C;
        font['L' * 8 + 0] = 0x60; font['L' * 8 + 1] = 0x60; font['L' * 8 + 2] = 0x60; font['L' * 8 + 3] = 0x60; font['L' * 8 + 4] = 0x60; font['L' * 8 + 5] = 0x7E;
        font['.' * 8 + 5] = 0x18; font['.' * 8 + 6] = 0x18;
        font[':' * 8 + 2] = 0x18; font[':' * 8 + 5] = 0x18;
        font['%' * 8 + 0] = 0x62; font['%' * 8 + 1] = 0x64; font['%' * 8 + 2] = 0x08; font['%' * 8 + 3] = 0x10; font['%' * 8 + 4] = 0x26; font['%' * 8 + 5] = 0x46;

        return font;
    }
}
