# Using dataloaders
Apart from loading data into dataframes, the `cesnet-datazoo` package provides dataloaders for processing data in smaller batches.

An example of how dataloaders can be used is in `cesnet_datazoo.datasets.loaders` or in the following snippet:

```python
def load_from_dataloader(dataloader: DataLoader):
    other_fields = []
    data_ppi = []
    data_flowstats = []
    labels = []
    for batch_other_fields, batch_ppi, batch_flowstats, batch_labels in dataloader:
        other_fields.append(batch_other_fields)
        data_ppi.append(batch_ppi)
        data_flowstats.append(batch_flowstats)
        labels.append(batch_labels)
    df_other_fields = pd.concat(other_fields, ignore_index=True)
    data_ppi = np.concatenate(data_ppi)
    data_flowstats = np.concatenate(data_flowstats)
    labels = np.concatenate(labels)
    return df_other_fields, data_ppi, data_flowstats, labels
```

When a dataloader is iterated, the returned data are in the format `tuple(batch_other_fields,  batch_ppi, batch_flowstats, batch_labels)`. Batch size *B* is configured with `batch_size` and `test_batch_size` config options.
The shapes are:

* batch_other_fields `pd.DataFrame (B, C)` - a Pandas DataFrame with [auxiliary fields][other-fields], such as communicating hosts, flow times, and more fields extracted from the ClientHello message. If the `return_other_fields` config option is false, this will be an empty DataFrame. Columns C depend on the used dataset and are available at `dataset_config.other_fields`.
* batch_ppi - `np.ndarray (B, [3, 4], 30)` - the middle dimension is either 4 when TCP push flags are used (`use_push_flags`) or 3 otherwise.
* batch_flowstats `np.ndarray (B, F)` - where F is the number of flowstats features computed with [DatasetConfig.get_flowstats_features_len][config.DatasetConfig.get_flowstats_features_len]. To get the order and names of flowstats features, call [DatasetConfig.get_flowstats_feature_names_expanded][config.DatasetConfig.get_flowstats_feature_names_expanded]. The batch_flowstats array includes flow statistics, TCP features (if available and configured), and bins of packet histograms (if available and configured). See the [data features][features] page for more information about features.
* batch_labels `np.ndarray (B)` - integer labels encoded with a `LabelEncoder` instance available at `dataset.class_info.encoder`.

PPI and flow statistics features returned from dataloaders are transformed depending on the selected configuration. See the [transforms][transforms] page for more information.