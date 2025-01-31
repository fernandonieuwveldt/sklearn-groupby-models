import numpy
import sklearn.base
import sklearn.model_selection

from collections import defaultdict


class BaseGroupbyModel(sklearn.base.BaseEstimator):
    """
    Splits data on group and apply model on each group
    """

    def __init__(self, estimator=None, split_feature=None):
        self.estimator = estimator
        self.split_estimators = defaultdict(list)
        self.split_feature = split_feature

    def fit(self, X=None, y=None):
        """
        Fits an estimator on each group and save

        :param X: pandas data_frame containing training data
        :param y: numpy array with model targets
        :return: BaseGroupbyModel object
        """
        for name, group_features in X.groupby(self.split_feature):
            group_target = y[group_features.index.values]
            estimator =  self.estimator(**self.kwargs)
            estimator.fit(group_features, group_target)
            self.split_estimators[name].append(estimator)
        return self

    def predict(self, X=None):
        """
        Apply estimator on each group splitted by split feature

        :param X: pandas dataframe with test data
        :return: numpy array of final predictions
        """
        predictions = numpy.zeros((X.shape[0], ))
        for name, group_features in X.groupby(self.split_feature):
            predictions[group_features.index.values] = self.split_estimators[name].predict(group_features)
        return predictions
