# Transforms
The `cesnet_datazoo` package supports configurable transforms of input data in a similar fashion to what torchvision is doing for the computer vision field. Input features are split into three groups, each having its own transformation. Those groups are PPI sequences, flow statistics, and packet histograms.

- Transformation configured in `ppi_transform` of [`DatasetConfig`][config.DatasetConfig] is applied to PPI sequences.
- `flowstats_transform` is applied to flow statistics (excluding boolean features, such as flow end reasons or TCP flags).
- `flowstats_phist_transform` is applied to packet histograms.

Transforms are implemented in a separate package [CESNET Models](https://github.com/CESNET/cesnet-models). See `cesnet_models.transforms` [documentation](https://cesnet.github.io/cesnet-models/reference_transforms/) for details.

!!! info "Limitations"
    The current implementation does not support the composing of transformations.

## Available transformations

**PPI sequences**

- [ClipAndScalePPI](https://cesnet.github.io/cesnet-models/reference_transforms/#transforms.ClipAndScalePPI)

**Flow statistics**

- [ClipAndScaleFlowstats](https://cesnet.github.io/cesnet-models/reference_transforms/#transforms.ClipAndScaleFlowstats)

**Packet histograms**

- [NormalizeHistograms](https://cesnet.github.io/cesnet-models/reference_transforms/#transforms.NormalizeHistograms)

More transformations will be implemented in future versions.

### Data scaling
Transformations implementing data scaling will be fitted, if needed, on a subset of training data during dataset initialization.