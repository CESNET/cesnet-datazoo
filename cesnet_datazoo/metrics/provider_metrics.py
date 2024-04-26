import numpy as np

from cesnet_datazoo.utils.class_info import ClassInfo


def provider_accuracies(true_labels: np.ndarray, preds: np.ndarray, class_info: ClassInfo) -> tuple[float, float]:
    provider_mapping_arr = np.array(list(class_info.provider_mapping.values()))
    true_labels_provider = provider_mapping_arr[true_labels]
    preds_provider = provider_mapping_arr[preds]
    mistakes = true_labels != preds
    provider_acc = (true_labels_provider == preds_provider).sum() / len(true_labels_provider)
    failed_provider_acc = (true_labels_provider[mistakes] == preds_provider[mistakes]).sum() / mistakes.sum()
    return provider_acc, failed_provider_acc

def per_app_provider_metrics(cm, class_info: ClassInfo):
    metrics = []
    for i, app in enumerate(class_info.target_names):
        if not class_info.has_provider[app]:
            metrics.append((None, None, None))
            continue
        group = class_info.group_matrix[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            sc_recall = cm[i, group].sum() / cm[i].sum()
            sc_precision = cm[group, i].sum() / cm[:, i].sum()
            sc_fscore = 2*sc_recall*sc_precision / (sc_recall + sc_precision)
        metrics.append((np.nan_to_num(sc_precision), np.nan_to_num(sc_recall), np.nan_to_num(sc_fscore)))
    return list(zip(*metrics))
