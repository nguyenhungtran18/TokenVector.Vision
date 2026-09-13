using System;
using System.Collections.Generic;
using TokenVector.Vision.Common;

namespace TokenVector.Vision.Transforms;

/// <summary>
/// Composable vision transform pipeline builder.
/// Supports fluent chaining and zero-allocation execution with buffer pooling.
/// </summary>
public sealed class ComposePipeline
{
    private readonly List<IVisionTransform> _transforms = [];

    public IReadOnlyList<IVisionTransform> Transforms => _transforms;

    public ComposePipeline Add(IVisionTransform transform)
    {
        ArgumentNullException.ThrowIfNull(transform);
        _transforms.Add(transform);
        return this;
    }

    public ComposePipeline Resize(int width, int height, InterpolationMode mode = InterpolationMode.Bilinear) =>
        Add(new ResizeTransform(width, height, mode));

    public ComposePipeline CenterCrop(int width, int height) =>
        Add(new CenterCropTransform(width, height));

    public ComposePipeline RandomCrop(int width, int height, int? seed = null) =>
        Add(new RandomCropTransform(width, height, seed));

    public ComposePipeline HorizontalFlip() =>
        Add(new HorizontalFlipTransform());

    public ComposePipeline VerticalFlip() =>
        Add(new VerticalFlipTransform());

    public ComposePipeline Rotate(float degrees) =>
        Add(new RotateTransform(degrees));

    public ComposePipeline ColorJitter(float brightness = 0.0f, float contrast = 0.0f, float saturation = 0.0f) =>
        Add(new ColorJitterTransform(brightness, contrast, saturation));

    public ComposePipeline Normalize(float[] mean, float[] std) =>
        Add(new NormalizeTransform(mean, std));

    public ComposePipeline NormalizeImageNet() =>
        Add(NormalizeTransform.ImageNet);

    public ComposePipeline ToCHW() =>
        Add(LayoutTransforms.ToCHW);

    /// <summary>
    /// Executes the pipeline sequentially across all registered transforms.
    /// </summary>
    public ImageBuffer Execute(ImageBuffer input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (_transforms.Count == 0) return input;

        ImageBuffer current = input;
        bool isFirst = true;

        for (int i = 0; i < _transforms.Count; i++)
        {
            var transform = _transforms[i];
            ImageBuffer next = transform.Apply(current);

            // Clean up intermediate temporary buffers (except input)
            if (!isFirst && current != input)
            {
                current.Dispose();
            }

            current = next;
            isFirst = false;
        }

        return current;
    }
}
