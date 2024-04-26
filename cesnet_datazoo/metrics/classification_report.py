import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from cesnet_datazoo.metrics.provider_metrics import per_app_provider_metrics, provider_accuracies
from cesnet_datazoo.utils.class_info import ClassInfo


def better_classification_report(test_labels: np.ndarray, preds: np.ndarray, cm: np.ndarray, labels: list[int], class_info: ClassInfo, digits: int = 2, zero_division: int = 0) -> tuple[str, dict[str, float]]:
    p, r, f1, s  = precision_recall_fscore_support(test_labels, preds,
                                                   labels=labels,
                                                   zero_division=zero_division)
    sc_p, sc_r, sc_f1 = per_app_provider_metrics(cm, class_info=class_info)
    predicted_unknown = cm[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted_unknown_perc = predicted_unknown / s
        predicted_unknown_perc = np.nan_to_num(predicted_unknown_perc)
    headers = ["precision (pr)", "recall (pr)", "f1-score (pr)", "pred unknown", "support"]
    headers_fmt = "{:>{width}} {:>15} {:>15} {:>15} {:>15} {:>9}\n"
    width = max(max(len(cn) for cn in class_info.target_names), len("failed provider acc"))
    report = headers_fmt.format("", *headers, width=width)
    report += "\n"
    row_fmt_provider = "{:>{width}} " + 3 * " {:>7.{digits}f} ({:.{digits}f}) " + " {:>7} ({:.{digits}f}) " + "{:>9}\n"
    row_fmt = "{:>{width}} " + 3 * " {:>7.{digits}f}        " + " {:>7} ({:.{digits}f}) " + "{:>9}\n"
    rows = zip(map(class_info.target_names.__getitem__, labels), p, sc_p, r, sc_r, f1, sc_f1, predicted_unknown, predicted_unknown_perc, s) # type: ignore
    for row in rows:
        app, p_, _, r_, _, f1_, _, u_, up_, s_ = row
        if class_info.has_provider[app]:
            report += row_fmt_provider.format(*row, width=width, digits=digits)
        else:
            report += row_fmt.format(app, p_, r_, f1_, u_, up_, s_, width=width, digits=digits)
    report += "\n"

    # Computing averages, ignoring the last element with metrics for the unknown class
    samples_sum = s.sum() # type: ignore
    predicted_unknown_sum = predicted_unknown.sum()
    avg_p, avg_r, avg_f1 = np.average(p[:-1]), np.average(r[:-1]), np.average(f1[:-1]) # type: ignore
    avg_sc_p = np.average(np.where(np.isnan([np.nan if x is None else x for x in sc_p]), p, sc_p)[:-1])
    avg_sc_r = np.average(np.where(np.isnan([np.nan if x is None else x for x in sc_r]), r, sc_r)[:-1])
    avg_sc_f1 = np.average(np.where(np.isnan([np.nan if x is None else x for x in sc_f1]), f1, sc_f1)[:-1])
    row_avg = [avg_p, avg_sc_p, avg_r, avg_sc_r, avg_f1, avg_sc_f1, predicted_unknown_sum, samples_sum]

    headers_avg = ["precision (pr)", "recall (pr)", "f1-score (pr)", "pred unknown", "support"]
    row_fmt_avg = "{:>{width}} " + 3 * " {:>6.{digits}} ({:.{digits}f}) " + "{:>15} " + "{:>9}\n"
    digits = 3 # show more precise averages
    report += headers_fmt.format("", *headers_avg, width=width)
    report += row_fmt_avg.format("macro avg", *row_avg, width=width, digits=digits)

    acc = accuracy_score(test_labels, preds)
    provider_acc, failed_provider_acc = provider_accuracies(test_labels, preds, class_info=class_info)

    row_fmt_acc = "{:>{width}} {:>15} {:>15} {:>7.{digits}f}\n"
    report += row_fmt_acc.format("acc", "", "", acc, width=width, digits=digits)
    report += row_fmt_acc.format("provider acc", "", "", provider_acc, width=width, digits=digits)
    report += row_fmt_acc.format("failed provider acc", "", "", failed_provider_acc, width=width, digits=digits)
    metrics = {
        "test/acc": acc,
        "test/provider-acc": provider_acc,
        "test/failed-provider-acc": failed_provider_acc,
        "test/fscore": avg_f1,
        "test/provider-fscore": avg_sc_f1,
        "test/recall": avg_r,
        "test/provider-recall": avg_sc_r,
    }
    return report, metrics
