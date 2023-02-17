from itertools import product

import numpy as np
import pandas as pd
import scipy

from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    multilabel_confusion_matrix, 
    accuracy_score, average_precision_score,
    roc_curve, roc_auc_score
)

from mstorch.evaluation.confusion_matrix import ConfusionMatrixDisplay


def _label_binarize(y_true, num_classes):

    if num_classes == 2:
        y_true_binarized = np.zeros((y_true.shape[0], 2), dtype=np.int32)
        mask = (y_true==1)
        y_true_binarized[~mask, 0] = 1
        y_true_binarized[mask, 1] = 1
    else:
        y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))

    return y_true_binarized


def compute_classification_metrics(y_true, y_proba, sample_weight=None, labels=None):

    num_classes = y_proba.shape[1]
    if labels is None:
        labels = np.arange(num_classes)

    y_true_binarized = _label_binarize(y_true, num_classes)
    y_pred = np.argmax(y_proba, axis=1)

    MCM = multilabel_confusion_matrix(
                y_true, y_pred, labels=labels, 
                sample_weight=sample_weight)

    tp_sum = MCM[:, 1, 1]
    fp_sum = MCM[:, 0, 1]
    fn_sum = MCM[:, 1, 0]
    tn_sum = MCM[:, 0, 0]

    # sensitivity, recall, hit rate, or true positive rate (TPR)
    recall = sensitivity = tp_sum / (tp_sum + fn_sum) 
    
    # specificity, selectivity or true negative rate (TNR)
    specificity = tn_sum / (tn_sum + fp_sum)
    
    # precision or positive predictive value (PPV)
    precision = ppv = tp_sum / (tp_sum + fp_sum)

    # negative predictive value (NPV)
    npv = tn_sum / (tn_sum + fn_sum)

    denom = precision + recall
    denom[denom == 0.] = 1  # avoid division by 0
    f_score = 2 * (precision * recall) / denom

    accuracy = tp_sum.sum()/y_true.shape[0]

    # Compute roc_auc, avg_precision, etc for each class
    roc_auc = np.zeros(len(labels))
    avg_precision = np.zeros(len(labels))
    for i, label in enumerate(labels):
        y_true_ = y_true_binarized[:, label]
        y_score_ = y_proba[:, label]
        roc_auc[i] = roc_auc_score(y_true_, y_score_)
        avg_precision[i] = average_precision_score(y_true_, y_score_)

    results = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f_score,
        'roc_auc': roc_auc,
        'average_precision': avg_precision
        # 'recall': recall,
        # 'precision': precision,
    }

    # per-experiment performance
    results2 = {
        'accuracy': accuracy
    }

    return results, results2


class ClassificationReport(object):
    
    def __init__(self, num_classes, display_labels=None):
        """Classification report generation.
        This is a helper class which can generate evaluation report for classification problems.
        Along with accuracy, sensitivity, specificity, roc_auc and more are computed per each class.

        Args:
            num_classes (int): number of classes
            display_labels (list, optional): Display labels. Defaults to None.
                If ``None``, the numerical index of the labels is used.
        """
        if display_labels:
            assert len(display_labels) == num_classes, "len(display_labels) should be equal to num_classes"
            self.display_labels = np.asarray(display_labels)
        else:
            self.display_labels = np.arange(num_classes)
        
        self.num_classes = num_classes
        self.y_trues = dict()
        self.y_scores = dict()
        self.sample_weights = dict()

    def add_experiment_result(self, exp_id, y_true, y_proba, sample_weight=None):
        """ Add classification experiment result to the report

        Args:
            exp_id ([string or numbers]): experiment ID
            y_true (numpy.array): array of true labels (n_samples,). Labels should be zero-based index.
            y_score (numpy.array): array of probability scores (n_samples, n_classes)
        """
        assert y_proba.shape[1] == self.num_classes
        assert y_true.shape[0] == y_proba.shape[0]
        if sample_weight is not None:
            assert (sample_weight.shape[0] == y_true.shape[0])

        self.y_trues[exp_id] = y_true
        self.y_scores[exp_id] = y_proba
        self.sample_weights[exp_id] = sample_weight
        
        return self

    def get_confusion_matrix_display(self, exp_id, labels=None):
        """Generate ConfusionMatrixDisplay instance for a classification experiment

        Args:
            exp_id ([string or numbers]): experiment ID
            labels (list): List of labels to index the matrix. This may be used to 
                reorder or select a subset of labels. If ``None``, the numerical 
                order of the all labels is used. Defaults to None.

        Returns:
            [ConfusionMatrixDisplay]: created instance of ConfusionMatrixDisplay
        """
        if labels is None:
            display_labels = self.display_labels
        else:
            display_labels = self.display_labels[labels]

        y_true = self.y_trues[exp_id]
        y_proba = self.y_scores[exp_id]
        y_pred = np.argmax(y_proba, axis=1)
        sample_weight = self.sample_weights[exp_id]

        cm_disp = ConfusionMatrixDisplay.create_instance(
                        y_true, y_pred, 
                        labels=labels,
                        sample_weight=sample_weight,
                        display_labels=display_labels)
        
        return cm_disp


    def get_roc_curve(self, exp_id, labels=None):
        
        y_true = self.y_trues[exp_id]
        y_proba = self.y_scores[exp_id]
        sample_weight = self.sample_weights[exp_id]
        y_true_bin = _label_binarize(y_true, self.num_classes)
        # linestyles = ['solid', 'dotted', 'dashed', 'dashdot']

        if labels is None:
            labels = list(range(self.num_classes))

        fig, ax = plt.subplots()
        legends = []
        for label_idx in labels:
            fpr, tpr, thresholds = roc_curve(y_true_bin[:, label_idx], y_proba[:, label_idx], sample_weight=sample_weight)
            roc_auc = roc_auc_score(y_true_bin[:, label_idx], y_proba[:, label_idx], sample_weight=sample_weight)
            ax.plot(
                fpr, 
                tpr, 
                # linestyle=linestyles[label_idx],
            )
            legends.append('class {} (area = {:.4f})'.format(self.display_labels[label_idx], roc_auc))

        ax.legend(legends, fontsize=12)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)

        return fig, ax


    def _compute_metrics(self, labels=None):

        if labels is None:
            display_labels = self.display_labels
        else:
            display_labels = self.display_labels[labels]

        results = dict() 
        # iterate over experiments
        for exp_id in self.y_trues.keys():
            y_true = self.y_trues[exp_id]
            y_proba = self.y_scores[exp_id]
            sample_weight = self.sample_weights[exp_id]
            report, report2 = compute_classification_metrics(
                                    y_true, y_proba, sample_weight=sample_weight, labels=labels)
            # init dictionary
            if len(results) == 0:
                results['per_class'] = dict()
                for k in list(report2):
                    results[k] = dict()

            results['per_class'][exp_id] = pd.DataFrame(report, index=display_labels)
            for k, v in report2.items():
                results[k][exp_id] = v

        return results, display_labels


    def _generate_tables(self, labels=None, confidence_level=0.95):

        # compute evaluation metrics for classification problem
        results, display_labels = self._compute_metrics(labels=labels)

        num_exp = len(results['accuracy'])
        assert num_exp > 0, 'No experiment results'

        # generate tables for mean and CI values for per-class performance metrics
        per_class_result = results['per_class']

        summary_df = pd.concat(per_class_result, axis=0)
        summary_arr = np.asarray([arr for arr in per_class_result.values()])
        z = scipy.stats.norm.ppf(0.5*(1+confidence_level))

        mean_arr = summary_arr.mean(axis=0)
        std_arr = summary_arr.std(axis=0)
        ci_arr = z * std_arr / np.sqrt(num_exp)
        upper_arr = mean_arr + ci_arr
        lower_arr = mean_arr - ci_arr

        upper_arr = upper_arr.clip(0., 1.)
        lower_arr = lower_arr.clip(0., 1.)

        columns = summary_df.columns
        mean_df = pd.DataFrame(mean_arr, index=display_labels, columns=columns)
        upper_df = pd.DataFrame(upper_arr, index=display_labels, columns=columns)
        lower_df = pd.DataFrame(lower_arr, index=display_labels, columns=columns)

        # generate table for accuracy and its CIs
        per_exp = {k:v for k, v in results.items() if k != 'per_class'}
        for metric_key, metric_val in per_exp.items():
            metric_arr = list(metric_val.values())
            metric_val['Mean'] = np.mean(metric_arr)
            metric_val['Std'] = np.std(metric_arr)
            ci = z * metric_val['Std'] / np.sqrt(len(metric_arr))
            metric_val['{:.0f}% CI'.format(confidence_level*100)] = ci

        accuracy_df = pd.DataFrame.from_dict(per_exp, orient='index').T

        return (accuracy_df, summary_df, mean_df, upper_df, lower_df)


    @staticmethod
    def combine_table(mean_df, upper_df, lower_df,
                        text_format='{avg:.1f} ({lb:.1f}-{ub:.1f})'):
        
        columns = [f'{col} (%)' for col in mean_df.columns]
        combine_df = pd.DataFrame(index=mean_df.index, columns=columns)
        for i, j in product(range(mean_df.shape[0]), range(mean_df.shape[1])):
            cell_text = text_format.format(
                            avg=mean_df.iloc[i, j]*100,
                            ub=upper_df.iloc[i, j]*100,
                            lb=lower_df.iloc[i, j]*100,
                        )
            combine_df.iloc[i, j] = cell_text

        return combine_df        

    def get_summary_tables(self, 
                           labels=None,
                           confidence_level=0.95,
                           text_format='{avg:.1f} ({lb:.1f}-{ub:.1f})'):

        accuracy_df, per_class_df, mean_df, upper_df, lower_df = \
            self._generate_tables(
                    labels=labels, 
                    confidence_level=confidence_level)

        # summary table for per-class performance metrics
        per_class_avg_df = self.combine_table(
                                    mean_df, upper_df, lower_df, 
                                    text_format=text_format)

        # accuracy table 
        accuracy_table_columns = []
        for col, sr in accuracy_df.iteritems():
            ci = sr.iloc[-1]
            sr[f'Mean ({sr.index[-1]})'] = text_format.format(
                avg=sr['Mean']*100, 
                lb=(sr['Mean']-ci)*100,
                ub=(sr['Mean']+ci)*100,
            )
            accuracy_table_columns.append(sr)
        accuracy_df = pd.concat(accuracy_table_columns, axis=1)

        return accuracy_df, per_class_df, per_class_avg_df


if __name__ == "__main__":
    
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    X, y = datasets.load_iris(return_X_y=True)
    mask = y < 2
    y = y[mask]
    X = X[mask, :]

    sample_weight = np.random.rand(y.shape[0])

    rpt = ClassificationReport(num_classes=2)
    clf = LogisticRegression().fit(X, y)
    y_proba = clf.predict_proba(X)
    rpt.add_experiment_result(1, y, y_proba, sample_weight=sample_weight)

    clf = DecisionTreeClassifier().fit(X, y)
    y_proba = clf.predict_proba(X)
    rpt.add_experiment_result(2, y, y_proba, sample_weight=sample_weight)

    clf = RandomForestClassifier().fit(X, y)
    y_proba = clf.predict_proba(X)
    rpt.add_experiment_result(3, y, y_proba, sample_weight=sample_weight)

    results = rpt.get_summary_tables(labels=[1])

    # conf_disp = rpt.get_confusion_matrix_display(1)
    # conf_disp.plot()
    # plt.savefig('./temp/conf_mat.jpg')

    # fig, ax = rpt.get_roc_curve(1, labels=[1])
    # plt.savefig('./temp/roc_curve.jpg')


