import numpy as np

from cesnet_datazoo.utils.class_info import ClassInfo


def superclass_accuracies(y_true: np.ndarray, y_pred: np.ndarray, class_info: ClassInfo) -> tuple[float, float]:
    y_true_sc = class_info.superclass_mapping_arr[y_true]
    y_pred_sc = class_info.superclass_mapping_arr[y_pred]
    mistakes = y_true != y_pred
    superclass_acc = (y_true_sc == y_pred_sc).sum() / len(y_true_sc)
    failed_superclass_acc = (y_true_sc[mistakes] == y_pred_sc[mistakes]).sum() / mistakes.sum()
    return superclass_acc, failed_superclass_acc

def per_app_superclass_metrics(cm, class_info: ClassInfo):
    metrics = []
    for i, app in enumerate(class_info.target_names):
        if not class_info.has_superclass[app]:
            metrics.append((None, None, None))
            continue
        group = class_info.group_matrix[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            sc_recall = cm[i, group].sum() / cm[i].sum()
            sc_precision = cm[group, i].sum() / cm[:, i].sum()
            sc_fscore = 2*sc_recall*sc_precision / (sc_recall + sc_precision)
        metrics.append((np.nan_to_num(sc_precision), np.nan_to_num(sc_recall), np.nan_to_num(sc_fscore)))
    return list(zip(*metrics))
