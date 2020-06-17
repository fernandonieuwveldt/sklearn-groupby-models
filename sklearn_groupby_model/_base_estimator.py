import numpy
import sklearn.base


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
        :return: model probabilities
        """
        predictions = numpy.zeros((X.shape[0], ))
        for name, group_features in X.groupby(self.split_feature):
            predictions[group.index.values] += self.split_estimators[name].predict(group_features)
        return predictions


class BaseGroupbyKFoldModel(sklearn.base.BaseEstimator):
    """
    Apply kfold split and apply model on each split
    """
class BaseKFoldModel():
    """
    Apply models on different fold splits of the data
    """

    def __init__(self, n_split=5, estimator=None, split_feature=None, **kwargs):
        self.n_split = n_split
        self.estimmator = None
        self.split_feature = split_feature
        self.split_estimators = defaultdict(list)
        self.kwargs = kwargs

    def fit(self, X=None, y=None):
        """
        Fits _SPLITS number of models
        :param X: pandas data_frame for training
        :param y: train target variable

        :return: BaseKFoldModel object
        """
        folds = KFold(n_splits=self._SPLITS, shuffle=True)
        for fold_n, (train_index, valid_index) in enumerate(folds.split(xtrain)):
            x_train_ = X[train_index, :]
            y_train_ = y[train_index]
            self.regressor.fit(x_train_, y_train_, **fit_params)
            self.estimators.append(self.Regressor)
        return self

    def predict(self, xtest=None):
        """
        Apply trained Regressor on test data
        :param xtest: Test data
        :return: array of predicted probability values
        """
        xtest = self.energy_transformer.transform(xtest)
        pred = numpy.zeros((xtest.shape[0], 2))
        for reg in self.estimators:
            pred += reg.predict(xtest)/self._SPLITS
        return pred
    
class GroupbyRegressorModel(sklearn.base.RegressorMixin):
    """
    Regressor based on model on different groupby splits
    """
    def __init__(self, estimator=None, split_feature=None):
        super().__init__(estimator=estimator,
                         split_feature=split_feature,
                         **kwargs
                         )


class GroupbyClassifierModel(sklearn.base.ClassifierMixin):
    """
    Classifier based on model on different groupby splits
    """
    def __init__(self, estimator=None, split_feature=None):
        super().__init__(estimator=estimator,
                         split_feature=split_feature,
                         **kwargs
                         )

class GroupbyPipeline:
    """
    Add classes to apply groupby transformers and estimators
    It can be a pipeline of transformers with the GroupbyEstimator at the end                                                                                                                                                                                                                                          
    """

