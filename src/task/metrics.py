from abc import ABCMeta, abstractmethod

import numpy as np


class BaseMetric(metaclass=ABCMeta):
    @abstractmethod
    def update(self, y_pred: np.ndarray, y_true: np.ndarray):
        pass

    @abstractmethod
    def compute(self) -> float:
        pass

    @abstractmethod
    def reset(self) -> float:
        pass


class AccuracyMetric(BaseMetric):
    def __init__(self):
        self.hits = 0
        self.total = 0

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Add batch
        """
        assert (
            y_pred.shape[0] == y_true.shape[0]
        ), f"Shape mismatch between y_pred ({y_pred.shape[0]}) and y_true ({y_true.shape[0]})"

        self.hits += (y_pred == y_true).sum().item()
        self.total += y_pred.shape[0]

    def compute(self):
        """
        Final metric
        """
        return self.hits / self.total

    def reset(self):
        self.hits = 0
        self.total = 0


def row_wise_isin(src: np.ndarray, trg: np.ndarray) -> np.ndarray:
    """
    Given two 2d arryas compute row-wise isin
    See comment here:
    https://stackoverflow.com/questions/67870579/rowwise-numpy-isin-for-2d-arrays

    >>> y_true = np.asarray([ [11457], [8740], [2779] ])
    >>> y_pred = np.asarray([ [2779, 8742], [8735, 5054], [ 299, 2779] ])
    >>> row_wise_isin(y_pred, y_true)
    array([[False, False],
           [False, False],
           [False,  True]])
    """
    return (src[:, :, None] == trg[:, None, :]).any(-1)


class RecallAtKMetric(BaseMetric):
    def __init__(self, k: int = 1):
        self.k = k
        self.hits = 0
        self.total = 0

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Add batch
        """

        assert (
            y_pred.shape[0] == y_true.shape[0]
        ), f"Shape mismatch between y_pred ({y_pred.shape[0]}) and y_true ({y_true.shape[0]})"

        assert (
            y_pred.shape[1] >= self.k
        ), f"`y_pred.shape[1]={y_pred.shape[1]}` must be >= recall@{self.k}"

        self.hits += row_wise_isin(y_pred[:, : self.k], y_true).any(-1).sum().item()
        self.total += y_pred.shape[0]

    def compute(self):
        """
        Final metric
        """
        return self.hits / self.total

    def reset(self):
        self.hits = 0
        self.total = 0
