# encoding=utf-8
"""
@author : zhongqing
"""
import numpy as np
from abc import ABCMeta, abstractmethod
import six
from base import BaseEstimator, MetaEstimatorMixin, clone
from _gradient import predict_stages
from decision_tree import  DecisionTreeRegressor


def _weighted_sum(sample_score, sample_weight, normalize=False):
    if normalize:
        return np.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()

def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    # Compute accuracy for each possible representation
    score = y_true == y_pred

    return _weighted_sum(score, sample_weight, normalize)
class ClassifierMixin(object):
    """Mixin class for all classifiers in scikit-learn."""
    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class BaseEnsemble(six.with_metaclass(ABCMeta, BaseEstimator,
                                      MetaEstimatorMixin)):
    @abstractmethod
    def __init__(self, base_estimator, n_estimators=10,
                 estimator_params=tuple()):
        # Set parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params
    def _validate_estimator(self, default=None):
        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

    def _make_estimator(self, append=True, random_state=None):

        estimator = clone(self.base_estimator_)
        #把参数展开，填充到新的 estimator中 .
        estimator.set_params(**dict((p, getattr(self, p))
                                    for p in self.estimator_params))

        if append:
            self.estimators_.append(estimator)
        return estimator

    def __len__(self):
        """Returns the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Returns the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Returns iterator over estimators in the ensemble."""
        return iter(self.estimators_)


class BaseGradientBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):
    @abstractmethod
    def __init__(self, loss, learning_rate, n_estimators, criterion,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 max_depth, min_impurity_decrease, min_impurity_split,
                 init, subsample, max_features,
                 random_state, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.presort = presort

    def _init_state(self):
        """Initialize model state and allocate model state data structures. """
        self.init_ = self.loss_.init_estimator()
        self.estimators_ = np.empty((self.n_estimators, ), dtype=np.object)
        self.train_score_ = np.zeros((self.n_estimators, ), dtype=np.float64)

        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators,), dtype=np.float64)



    def _fit_stage(self, i, X, y, y_pred, sample_weight, sample_mask,
                   random_state, X_idx_sorted, X_csc=None, X_csr=None):

        loss = self.loss_
        original_y = y

        # 参考 loss.py ,  y - expit(y_pred.ravel())
        residual = loss.negative_gradient(y, y_pred)

        # induce regression tree on residuals

        tree = DecisionTreeRegressor(
            criterion=self.criterion,
            splitter='best',
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=random_state,
            presort=self.presort)

        tree.fit(X, residual)
        # 这个函数会更新  y_pred .
        loss.update_terminal_regions(tree.tree_, X, y, residual, y_pred,
                                     sample_weight, sample_mask,
                                     self.learning_rate, k=0)

        self.estimators_[i] = tree

        return y_pred

    def fit(self, X, y, sample_weight=None, monitor=None):
        n_samples, self.n_features_ = X.shape
        if sample_weight is None:
            # np.ones 就是全1 的意思.
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            raise NotImplementedError("Don't consider sample weight ")


        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model - FIXME make sample_weight optional
            self.init_.fit(X, y, sample_weight)

            # init predictions
            y_pred = self.init_.predict(X) # y_pred的初始值 ！！ .
            begin_at_stage = 0
        else:

            raise NotImplementedError("Don't consider warm-start")
            begin_at_stage = self.estimators_.shape[0]
            y_pred = self._decision_function(X)
            self._resize_state()

        X_idx_sorted = None
        presort = self.presort

        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        # 这是一个简单优化，方便快速运算,避免多次重复sort.
        if presort == 'auto' and issparse(X):
            presort = False
        elif presort == 'auto':
            presort = True

        if presort == True:

            X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),dtype=np.int32)

        n_stages = self._fit_stages(X, y, y_pred, sample_weight, random_state,
                                    begin_at_stage, monitor, X_idx_sorted)

        # change shape of arrays after fit (early-stopping or additional ests)
        # 处理early-stopping, 比如说设定了100棵，可能只算出了 56棵就足够.
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        return self

    def _fit_stages(self, X, y, y_pred, sample_weight, random_state,
                    begin_at_stage=0, monitor=None, X_idx_sorted=None):
        """Iteratively fits the stages.
               For each stage it computes the progress (OOB, train score)
               and delegates to ``_fit_stage``.
               Returns the number of stages fit; might differ from ``n_estimators``
               due to early stopping.
               """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=np.bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_
        # 打印信息.
        # if self.verbose:
        #     verbose_reporter = VerboseReporter(self.verbose)
        #     verbose_reporter.init(self, begin_at_stage)

        # perform boosting iterations
        i = begin_at_stage

        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag,
                                                  random_state)

                old_oob_score = loss_(y[~sample_mask],
                                      y_pred[~sample_mask],
                                      sample_weight[~sample_mask])

            # fit next stage of trees
            y_pred = self._fit_stage(i, X, y, y_pred, sample_weight,
                                     sample_mask, random_state, X_idx_sorted)

            if do_oob:
                self.train_score_[i] = loss_(y[sample_mask],
                                             y_pred[sample_mask],
                                             sample_weight[sample_mask])
                self.oob_improvement_[i] = (
                    old_oob_score - loss_(y[~sample_mask],
                                          y_pred[~sample_mask],
                                          sample_weight[~sample_mask]))

            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y, y_pred, sample_weight)

            ## 这里可以自定义实现early_stopping.
            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break
        return i +1

    def _init_decision_function(self, X):
        """Check input and compute prediction of ``init``. """
        score = self.init_.predict(X).astype(np.float64)
        return score

    def _decision_function(self, X):
        # for use in inner loop, not raveling the output in single-class case,
        # not doing input validation.
        score = self._init_decision_function(X)
        predict_stages(self.estimators_, X, self.learning_rate, score)
        return score



    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : array, shape = [n_features]

        注意：这里的len(stage)是什么意思?
        """

        total_sum = np.zeros((self.n_features_,), dtype=np.float64)
        for stage in self.estimators_:
            stage_sum = sum(tree.feature_importances_
                            for tree in stage) / len(stage)
            total_sum += stage_sum

        importances = total_sum / len(self.estimators_)
        return importances

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indic

        """
        n_estimators, _ = self.estimators_.shape

        leaves = np.zeros((X.shape[0], n_estimators))
        for i in range(n_estimators):
            estimator = self.estimators_[i]
            leaves[:, i] = estimator.apply(X)

        return leaves





def _random_sample_mask(n_total_samples, n_total_in_bag, random_state):
    """
    随机选出 n_total_in_bag个元素.
    :param n_total_samples:
    :param n_total_in_bag:
    :param random_state:
    :return:
    """
    rand_vals = random_state.rand(n_total_samples)
    sample_mask = np.zeros((n_total_samples, ))
    n_bagged = 0
    for i in range(n_total_samples):
        if rand_vals[i] * (n_total_samples - i ) < n_total_in_bag - n_bagged:
            sample_mask[i] = 1
            n_bagged += 1
    return sample_mask



class GradientBoostingClassifier(BaseGradientBoosting, ClassifierMixin):
    _SUPPORTED_LOSS = ('deviance')

    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto'):
        super(GradientBoostingClassifier, self).__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start,
            presort=presort)

    def decision_function(self, X):
        score = self._decision_function(X)
        return score.ravel()

    def predict(self, X):
        """
        p =  expit(score）
        :param X:
        :return:
        """
        score = self.decision_function(X)
        decisions = self.loss_._score_to_decision(score)

        return self.classes_.take(decisions, axis=0)



    def predict_proba(self, X):
         score = self.decision_function(X)
         return self.loss_._score_to_proba(score)





