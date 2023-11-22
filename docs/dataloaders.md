# Using dataloaders
Apart from loading data into dataframes, the `cesnet-datazoo` package provides dataloaders for processing data in smaller batches.

An example of how dataloaders can be used is in `cesnet_datazoo.datasets.loaders` or in the following snippet:

```python
def load_from_dataloader(dataloader: DataLoader):
    data_ppi = []
    data_flowstats = []
    labels = []
    for batch_ppi, batch_flowstats, batch_labels in dataloader:
        data_ppi.append(batch_ppi)
        data_flowstats.append(batch_flowstats)
        labels.append(batch_labels)
    data_ppi = np.concatenate(data_ppi)
    data_flowstats = np.concatenate(data_flowstats)
    labels = np.concatenate(labels)
    return data_ppi, data_flowstats, labels

```

When a dataloader is iterated, the returned data are in the format `tuple(batch_ppi, batch_flowstats, batch_labels)`. The batch size *B* is configured with `batch_size` and `test_batch_size` config options.
The shapes are:

* batch_ppi - `(B, [3, 4], 30)` - the middle dimension is either 4 when TCP push flags are used (`use_push_flags`) or 3 otherwise.
* batch_flowstats `(B, F)` - where F is the number of flowstats features computed with [DatasetConfig.get_flowstats_features_len][config.DatasetConfig.get_flowstats_features_len]. To get the order and names of flowstats features, call [DatasetConfig.get_flowstats_feature_names_expanded][config.DatasetConfig.get_flowstats_feature_names_expanded]. The batch_flowstats array includes flow statistics, TCP features (if available and configured), and bins of packet histograms (if available and configured). See the [data features][features] page for more information about features.
* batch_labels `(B)` - integer labels encoded with `LabelEncoder` available at `dataset.encoder`.

Data returned from dataloaders are scaled depending on the selected configuration; see [`DatasetConfig`][config.DatasetConfig] for options.