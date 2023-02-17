from abc import ABC, abstractmethod


class BaseTask(ABC):
    """
    An abstract base class for a machine learning task
    """
    def __init__(self, name, label_column, prediction_column):
        self.name = name
        self.label_column = label_column
        self.prediction_column = prediction_column
    
    @classmethod
    def is_regression(cls):
        return issubclass(cls, RegressionTask)

    @classmethod
    def is_classification(cls):
        return issubclass(cls, ClassificationTask)

    @classmethod
    def is_object_detection(cls):
        return issubclass(cls, ObjectDetectionTask)

    @classmethod
    def is_segmentation(cls):
        return issubclass(cls, SegmentationTask)

    @classmethod
    def is_multitask(cls):
        return issubclass(cls, MultiTask)

    def __key(self):
        return self.name

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, BaseTask):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(`{self.name}`)'

    @property
    def num_tasks(self):
        return 1

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.num_tasks:
            raise StopIteration

        if self.is_multitask():
            next_task = self[self.index]
            self.index += 1
            return next_task
        else:
            self.index += 1
            return self


class ClassificationTask(BaseTask):
    
    def __init__(self, 
                 name,
                 label_column, 
                 prediction_column, 
                 num_classes, 
                 label_decode_map=None):
        super().__init__(name, label_column, prediction_column)
        self.num_classes = num_classes
        if label_decode_map:
            self.label_decode_map = label_decode_map
        else:
            self.label_decode_map = list(range(num_classes))

    @property
    def labels(self):
        return list(self.label_decode_map)


class RegressionTask(BaseTask):
    def __init__(self, name, label_column, prediction_column):
        super().__init__(name, label_column, prediction_column)


class ObjectDetectionTask(BaseTask):
    def __init__(self, 
                name, 
                label_column='labels',
                box_column='boxes',
                score_column='scores',
                num_classes=2,
                box_dim=2):
        super().__init__(name, label_column, None)
        assert box_dim in (1, 2)
        self.box_column = box_column
        self.score_column = score_column
        self.box_dim = box_dim
        self.num_classes = num_classes

class SegmentationTask(BaseTask):
    pass



class MultiTask(BaseTask):

    def __init__(self, tasks):
        assert all(isinstance(t, BaseTask) for t in tasks)
        self._tasks = {t.name: t for t in tasks}
        task_names = list(self._tasks)
        assert len(set(task_names)) == len(task_names)
        name = '-'.join(['multitask'] + task_names)
        super().__init__(name, label_column=None, prediction_column=None)

    def add(self, task):
        assert isinstance(task, BaseTask)
        self._task[task] = task

    @property
    def tasks(self):
        return list(self._tasks.values())

    @property
    def names(self):
        return list(self._tasks)

    @property
    def num_tasks(self):
        return len(self.tasks)

    def __getitem__(self, index):
        return self.tasks[index]

    def find_by_name(self, name):
        return self._tasks[name]

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.tasks:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
