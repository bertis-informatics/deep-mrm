from itertools import product

import numbers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# from sklearn.metrics import confusion_matrix
# y_true = [2, 0, 2, 2, 0, 1]
# y_pred = [0, 0, 2, 2, 0, 2]
# weight = [0.5, 0.1, 0.8, 1.3, 0.1, 0.9]
# cm= confusion_matrix(y_true, y_pred, sample_weight=weight)

class ConfusionMatrixDisplay(object):

    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    @property
    def num_classes(self):
        return self.confusion_matrix.shape[0]

    @classmethod
    def create_instance(cls, y_true, y_pred, labels, sample_weight=None, display_labels=None):
        cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
        if display_labels is None:
            display_labels = list(labels)
        return cls(cm, display_labels)

    def plot(self, 
             normalize='true',
             xticks_rotation='horizontal',
             cmap=plt.cm.Blues, 
             ax=None, 
             colorbar=True):
        
        # when sample_weight is given, confusion matrix may have real numbers
        if issubclass(self.confusion_matrix.dtype.type,  numbers.Integral):
            text_format = '{cell_cnt:d} ({cell_pct:.0f}%)'
        else:
            text_format = '{cell_cnt:.2f} ({cell_pct:.0f}%)'

        if normalize not in ['true', 'pred', 'all']:
            raise ValueError(
                    "normalize must be one of {'true', 'pred', 'all'}")
        
        cm = self.confusion_matrix
        with np.errstate(all='ignore'):
            if normalize == 'true':
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif normalize == 'pred':
                cm = cm / cm.sum(axis=0, keepdims=True)
            elif normalize == 'all':
                cm = cm / cm.sum()
            cm = np.nan_to_num(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1.0)
        thresh = (cm.max() + cm.min()) / 2.0
        self.text = np.empty_like(cm, dtype=object)

        for i, j in product(range(self.num_classes), range(self.num_classes)):
            color = "white" if cm[i, j] > thresh else "black"
            text_cm = text_format.format(
                            cell_cnt=self.confusion_matrix[i, j],
                            cell_pct=cm[i, j]*100)
            self.text[i, j] = ax.text(
                                j, i, text_cm,
                                ha="center", va="center",
                                color=color, fontsize=8)
        if colorbar:
            ticks = np.linspace(0, 1.0, 6, endpoint=True)
            cbar = fig.colorbar(im, ticks=ticks, format='%.1f')
            # cbar.ax.set_yticklabels(tick_labels)
        
        ax.set(xticks=np.arange(self.num_classes),
            yticks=np.arange(self.num_classes),
            xticklabels=self.display_labels,
            yticklabels=self.display_labels,
            ylabel="True class",
            xlabel="Predicted class")        
        
        ax.set_ylim((self.num_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure = fig
        self.ax = ax
        self.im = im

        return self


def plot_confusion_matrix(y_true, 
                          y_pred, 
                          labels,
                          normalize=False,
                          title=None,
                          xticks_rotation='horizontal',
                          colorbar=True,
                          cmap=plt.cm.Blues):
    """Plot and print confusion matrix from the input data
        
    Args:
        y_true (Array): Ground truth target values
        y_pred (Array): Estimated targets as returned by a classifier
        labels (list or dict): Class labels of input data. This may be used to reorder labels
        normalize (bool)(opt): Determine whether the matrix shows normalized values or real values
        title (str)(opt): Title of the plot image
        cmap (Matplotlib cmap)(opt): Color map style of the confusion matrix

    Returns:
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    
    if isinstance(labels, dict):
        class_indexes = [class_idx for class_idx, class_name in labels.items()]
        class_names = [class_name for class_idx, class_name in labels.items()]
    else:
        class_indexes = list(labels)
        class_names = list(labels)

    cm = confusion_matrix(y_true, y_pred, labels=class_indexes)
    n_classes = cm.shape[0]
    cm1 = cm

    if normalize:
        cm = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100,0)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)    
    thresh = (cm.max() + cm.min()) / 2.0
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # color = cmap_max if cm[i, j] < thresh else cmap_min
            color = "white" if cm[i, j] > thresh else "black"
            num = format(cm[i, j], fmt)
            num = num.split(".")[0]
            text_cm = str(cm1[i, j]) + " (" + num+"%)"
            ax.text(
                j, i, text_cm,
                ha="center", va="center",
                color=color, fontsize=8)

    if colorbar:
        ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel="True class",
           xlabel="Predicted class")        
    
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    # fig.tight_layout()
    
    return fig, ax

def test():
    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    cm_disp = ConfusionMatrixDisplay.create_instance(
                    y_test, y_pred, 
                    np.arange(len(class_names)), 
                    display_labels=class_names
                )
    cm_disp = cm_disp.plot(cmap=plt.cm.Blues)
    plt.savefig("confusion_matrix.png")

    return
