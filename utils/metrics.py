import numpy as np


class SegMetric(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """
        Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        FP = hist.sum(axis=0) - np.diag(hist)
        FN = hist.sum(axis=1) - np.diag(hist)
        Original_TP = np.diag(hist)
        TP = Original_TP.copy()
        TP[TP == 0] = 1
        # TN = hist.sum() - (FP + FN + TP)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        f1 = (2 * (precision*recall) / (precision + recall)).mean()

        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) +
                              hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
                "Overall F1: \t": f1
            },
            cls_iu,
        )
        
    def get_f1_score(self):
        hist = self.confusion_matrix
        FP = hist.sum(axis=0) - np.diag(hist)
        FN = hist.sum(axis=1) - np.diag(hist)
        Original_TP = np.diag(hist)
        TP = Original_TP.copy()
        TP[TP == 0] = 1
        # TN = hist.sum() - (FP + FN + TP)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        f1 = (2 * (precision*recall) / (precision + recall)).mean()
        return f1

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))