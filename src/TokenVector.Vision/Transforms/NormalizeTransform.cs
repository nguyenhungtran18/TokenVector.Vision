using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Transforms;

/// <summary>
/// Vectorized Normalization Transform: (x - mean) / std or (x * scale) + bias.
/// Accelerated via AVX2 and FMA SIMD intrinsics.
/// </summary>
public sealed unsafe class NormalizeTransform : IVisionTransform
{
    private readonly float[] _mean;
    private readonly float[] _std;
    private readonly float[] _invStd;

    /// <summary>
    /// Default ImageNet normalization: Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]
    /// </summary>
    public static NormalizeTransform ImageNet { get; } = new([0.485f, 0.456f, 0.406f], [0.229f, 0.224f, 0.225f]);

    public NormalizeTransform(float[] mean, float[] std)
    {
        ArgumentNullException.ThrowIfNull(mean);
        ArgumentNullException.ThrowIfNull(std);
        if (mean.Length != std.Length)
            throw new ArgumentException("Mean and Std arrays must have identical length.");

        _mean = (float[])mean.Clone();
        _std = (float[])std.Clone();
        _invStd = new float[std.Length];

        for (int i = 0; i < std.Length; i++)
        {
            if (std[i] == 0) throw new DivideByZeroException($"Std value for channel {i} cannot be zero.");
            _invStd[i] = 1.0f / std[i];
        }
    }

    public ImageInfo GetOutputInfo(ImageInfo inputInfo) =>
        new(inputInfo.Width, inputInfo.Height, inputInfo.Channels, 
            inputInfo.Layout == MemoryLayout.CHW ? PixelFormat.Float32Planar : PixelFormat.Float32Rgb,
            0, inputInfo.Layout);

    public ImageBuffer Apply(ImageBuffer input, ImageBuffer? output = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var outInfo = GetOutputInfo(input.Info);
        var target = output ?? new ImageBuffer(outInfo);

        int channels = input.Channels;
        int totalPixels = input.Width * input.Height;

        fixed (float* meanPtr = _mean)
        fixed (float* invStdPtr = _invStd)
        {
            if (input.Layout == MemoryLayout.CHW)
            {
                // Planar normalization: each channel is contiguous
                for (int c = 0; c < channels; c++)
                {
                    float m = meanPtr[c % _mean.Length];
                    float invS = invStdPtr[c % _invStd.Length];

                    float* srcPlane = (float*)input.GetChannelPlaneFloatPointer(c);
                    float* dstPlane = target.GetChannelPlaneFloatPointer(c);

                    int i = 0;
                    if (Avx.IsSupported)
                    {
                        var vMean = Vector256.Create(m);
                        var vInvStd = Vector256.Create(invS);

                        for (; i <= totalPixels - 8; i += 8)
                        {
                            var vSrc = Avx.LoadVector256(srcPlane + i);
                            var vSub = Avx.Subtract(vSrc, vMean);
                            var vResult = Avx.Multiply(vSub, vInvStd);
                            Avx.Store(dstPlane + i, vResult);
                        }
                    }

                    for (; i < totalPixels; i++)
                    {
                        dstPlane[i] = (srcPlane[i] - m) * invS;
                    }
                }
            }
            else
            {
                // Interleaved normalization HWC
                if (!input.IsFloatingPoint)
                {
                    byte* src = input.BytePointer;
                    float* dst = target.FloatPointer;

                    for (int i = 0; i < totalPixels; i++)
                    {
                        int offset = i * channels;
                        for (int c = 0; c < channels; c++)
                        {
                            float normVal = src[offset + c] * (1.0f / 255.0f);
                            dst[offset + c] = (normVal - meanPtr[c % _mean.Length]) * invStdPtr[c % _invStd.Length];
                        }
                    }
                }
                else
                {
                    float* src = input.FloatPointer;
                    float* dst = target.FloatPointer;

                    for (int i = 0; i < totalPixels; i++)
                    {
                        int offset = i * channels;
                        for (int c = 0; c < channels; c++)
                        {
                            dst[offset + c] = (src[offset + c] - meanPtr[c % _mean.Length]) * invStdPtr[c % _invStd.Length];
                        }
                    }
                }
            }
        }

        return target;
    }
}
