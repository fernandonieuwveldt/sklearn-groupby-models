import sklearn.base


class BaseGroupbyTransformer(sklearn.base.TransformerMixin):
    """
    This class groups data by split_feature and apply a transformer on each group. Can be used in an sklearn
    pipeline
    """
    def __init__(self, split_feature=None, transformer=None, **kwargs):
        self.split_feature = split_feature
        self.transformer = transformer
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        Fit transformer on each group of data
        
        """
        self.group_transform_ = {}
        for _id, group in X.groupby(self.split_feature):
            self.group_transform_[_id] = self.transformer(**self.kwargs).fit(X, y)
        return self

    def transform(self, X, y=None):
        for _id, group_features in X.groupby(self.split_feature):
            X[group_features.index.values, :] = self.group_transform_[_id].transform(X[group_indices, :])
        return X
