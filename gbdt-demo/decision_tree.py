# encoding=utf-8
"""
@author : zhongqing
"""
import six
from abc import ABCMeta, abstractmethod
from base import BaseEstimator
import numpy as np
from _tree import  DepthFirstTreeBuilder, Tree
def r2_score(y , y_pred):
    raise NotImplemented("check R^2 wiki. easy")

class RegressorMixin(object):
    """Mixin class for all regression estimators in scikit-learn."""
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction."""
        return r2_score(y, self.predict(X))



class BaseDecisionTree(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for decision trees.
       Warning: This class should not be used directly.
       Use derived classes instead.
       """

    @abstractmethod
    def __init__(self,
                 criterion,
                 splitter,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 min_weight_fraction_leaf,
                 max_features,
                 max_leaf_nodes,
                 random_state,
                 min_impurity_decrease,
                 min_impurity_split,
                 class_weight=None,
                 presort=False):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        n_samples, self.n_features_ = X.shape
        y = np.atleast_1d(y)
        expanded_class_weight = None

        # is_classification = is_classifier(self)
        # if is_classification:
        #     y = np.copy(y)
        #     y_encoded = np.zeros(y.shape, dtype=np.int)
        #     classes_k , y_encoded = np.unique( y , return_inverse=True)
        #     y = y_encoded
        # else:
        self.classes_ = [None]

        self.n_classes_ = [1]


        # Build tree

        criterion = self.criterion
        splitter = self.splitter


        self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)

        builder = DepthFirstTreeBuilder(splitter, self.min_samples_split,
                                            self.min_samples_leaf,
                                            self.min_weight_leaf,
                                            self.max_depth,
                                            self.min_impurity_decrease,
                                            self.min_impurity_split)

        builder.build(self.tree_, X, y, sample_weight, X_idx_sorted)

        return self

    def predict(self, X, check_input=True):
        proba = self.tree_.predict(X)
        return proba[:, 0]

    def apply(self, X, check_input=True):
        return self.tree_.apply(X)

    @property
    def feature_importances_(self):
        """Return the feature importances.
        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """

        return self.tree_.compute_feature_importances()



class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    def __init__(self,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 presort=False):
        super(DecisionTreeRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            presort=presort)

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        super(DecisionTreeRegressor, self).fit(
            X, y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted)
        return self
