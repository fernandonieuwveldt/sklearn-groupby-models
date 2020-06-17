import sklearn.base


class BaseGroupbyTransformer(skearn.base.TransformerMixin):
    """
    This class groups data by feature and apply a transformer on each group. Can be used in an sklearn
    pipeline
    """
    def __init__(self, feature=None, transformer=None, **kwargs):
        self.feature = feature
        self.transformer = transformer
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        Fit transformer on each group of data
        
        """
        self.group_transform_ = {}
        for _id, group in X.groupby(self.feature):
            self.group_transform_[_id] = self.transformer(**self.kwargs).fit(X, y)
        return self

    def transform(self, X, y=None):
        for _id, group in X.groupby(self.feature):
            group_indices = X[self.feature]==_id
            X[group_indices, :] = self.group_transform_[_id].transform(X[group_indices, :])
        return X
