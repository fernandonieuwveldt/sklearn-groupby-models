import numpy
import sklearn.base
from _base_estimator import BaseGroupbyModel


class GroupbyClassifierModel(BaseGroupbyModel, sklearn.base.ClassifierMixin):
    """
    Classifier based model on different groupby splits
    """
    def __init__(self, estimator=None, split_feature=None, **kwargs):
        super().__init__(estimator=estimator,
                         split_feature=split_feature,
                         **kwargs)

    def predict_proba(self, X=None):
        """
        Apply estimator on each group splitted by split feature and compute probabilities

        :param X: pandas dataframe with test data
        :return: numpy array of probabilities
        """
        predictions = numpy.zeros((X.shape[0], ))
        for name, group_features in X.groupby(self.split_feature):
            predictions[group_features.index.values] = self.split_estimators[name].predict_proba(group_features)
        return predictions


class GroupbyRegressorModel(BaseGroupbyModel, sklearn.base.RegressorMixin):
    """
    Regressor based model on different groupby splits
    """
    def __init__(self, estimator=None, split_feature=None):
        super().__init__(estimator=estimator,
                         split_feature=split_feature,
                         **kwargs)


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    clf = GroupbyClassifierModel(estimator=LogisticRegression())
