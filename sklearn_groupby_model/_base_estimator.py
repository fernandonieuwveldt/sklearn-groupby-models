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
            group_target = y[group.index.values]
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
            predictions[group.index.values] += self.split_estimators[name].predict(group_features)
        return predictions


    
class GroupbyRegressorModel(BaseGroupbyModel, sklearn.base.RegressorMixin):
    """
    Regressor based on model on different groupby splits
    """
    def __init__(self, estimator=None, split_feature=None):
        super().__init__(estimator=estimator,
                         split_feature=split_feature,
                         **kwargs)


class GroupbyClassifierModel(BaseGroupbyModel, sklearn.base.ClassifierMixin):
    """
    Classifier based on model on different groupby splits
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
            predictions[group.index.values] += self.split_estimators[name].predict_proba(group_features)
        return predictions

class GroupbyPipeline:
    """
    Add classes to apply groupby transformers and estimators
    It can be a pipeline of transformers with the GroupbyEstimator at the end                                                                                                                                                                                                                                          
    """


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    clf = GroupbyClassifierModel(estimator=LogisticRegression())
