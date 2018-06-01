# encoding=utf-8
"""
@author : zhongqing 
"""
import six
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import expit
TREE_LEAF = -1


class LossFunction(six.with_metaclass(ABCMeta, object)):
    """Abstract base class for various loss functions."""

    def __init__(self, n_classes=1):
        self.K = n_classes

    def init_estimator(self):
        """Default ``init`` estimator for loss function. """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, y, pred, sample_weight=None):
        """Compute the loss of prediction ``pred`` and ``y``. """

    @abstractmethod
    def negative_gradient(self, y, y_pred, **kargs):
        """Compute the negative gradient.
        Parameters
        ---------
        y : np.ndarray, shape=(n,)
            The target labels.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """
    def update_terminal_regions(self, tree, X, y, residual, y_pred,sample_weight, sample_mask, learning_rate=1.0, k=0):
        """
        Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.
        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray, shape=(n, m)
            The data array.
        y : ndarray, shape=(n,)
            The target labels.
        residual : ndarray, shape=(n,)
            The residuals (usually the negative gradient).
        y_pred : ndarray, shape=(n,)
            The predictions.
        sample_weight : ndarray, shape=(n,)
            The weight of each sample.
        sample_mask : ndarray, shape=(n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default 0
            The index of the estimator being updated.
            针对multi-class的情况，目前只考虑binary class，这里就是k=0
        """

        terminal_regions = tree.apply(X)


        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # np.where 返回的是 tuple  (  inddicse,    ) ,用  [0] 才能获取真正的 indices，也就是 叶子节点对应的node_id
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions,
                                         leaf, X, y, residual,
                                         y_pred[:, k], sample_weight)

        # update predictions (both in-bag and out-of-bag)
        y_pred  += (learning_rate
                         * tree.value.take(terminal_regions, axis=0))
    @abstractmethod
    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Template method for updating terminal regions (=leaves). """


class ClassificationLossFunction(six.with_metaclass(ABCMeta, LossFunction)):
    """Base class for classification loss functions. """

    def _score_to_proba(self, score):
        """Template method to convert scores to probabilities.
         the does not support probabilities raises AttributeError.
        """
        raise TypeError('%s does not support predict_proba' % type(self).__name__)

    @abstractmethod
    def _score_to_decision(self, score):
        """Template method to convert scores to decisions.
        Returns int arrays.
         """

class LogOddsEstimator(object):
    """
    An estimator predicting the log odds ratio.

    这里的odds指： 正样本个数/负样本个数

    odds  =  p/(1-p)  =  e^pred
    logodds = pred
    """
    scale = 1.0

    def fit(self, X, y, sample_weight=None):
        # pre-cond: pos, neg are encoded as 1, 0
        if sample_weight is None:
            pos = np.sum(y)
            neg = y.shape[0] - pos
        else:
            raise NotImplementedError()

        if neg == 0 or pos == 0:
            raise ValueError('y contains non binary labels.')
        self.prior = self.scale * np.log(pos / neg)

    def predict(self, X):
        # check_is_fitted(self, 'prior')

        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(self.prior)
        return y



class BinomialDeviance(ClassificationLossFunction):
    """Binomial deviance loss function for binary classification.
       Binary classification is a special case; here, we only need to
       fit one tree instead of ``n_classes`` trees.
       """

    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes.".format(
                self.__class__.__name__))
        # we only need to fit one tree for binary clf.
        super(BinomialDeviance, self).__init__(1)

    def init_estimator(self):
        return LogOddsEstimator()

    def __call__(self, y, pred, sample_weight=None):
        """
        Compute the deviance (= 2 * negative log-likelihood).
         这里的

         p = e^pred / ( 1 + e^pred )
         1 - p =  1/ (1 + e^pred )

        """
        # logaddexp(0, v) == log(1.0 + exp(v))
        pred = pred.ravel()
        if sample_weight is None:
            return -2.0 * np.mean((y * pred) - np.logaddexp(0.0, pred))
        else:
            raise NotImplementedError("demo ,not implemented")
    def negative_gradient(self, y, pred, **kargs):
        """Compute the residual (= negative gradient).
        对上面的 __call__()中的函数求梯度.
        The expit function, also known as the logistic function, is defined as
         expit(x) = 1/(1+exp(-x)). It is the inverse of the logit function.

        """
        return y - expit(pred.ravel())

    def _score_to_proba(self, score):
        """
        这里输出了 一个  N * 2 的矩阵，包括了 正负类的概率
        :param score:
        :return:
        """
        proba = np.ones((score.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(score.ravel())
        proba[:, 0] -= proba[:, 1]
        return proba

    def _score_to_decision(self, score):
        proba = self._score_to_proba(score)
        return np.argmax(proba, axis=1)
    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Make a single Newton-Raphson step.
                our node estimate is given by:
                    sum(w * (y - prob)) / sum(w * prob * (1 - prob))
                we take advantage that: y - prob = residual

        每个leaf 代表一个 terminal_region .
        """

        #注意: terminal_regions  =tree.apply(X) ， 表示 每个样本对应的叶子节点编号
        # np.where(terminal_regions == leaf)[0]就是本次只处理  落在leaf的 样本集
        terminal_region = np.where(terminal_regions == leaf)[0]
        #根据索引，取出residual值.
        residual = residual.take(terminal_region, axis=0)
        #根据索引，取出y值
        y = y.take(terminal_region, axis=0)
        #取出样本权重，我们可以认为是全1 ；一般不需要特别设置.
        sample_weight = sample_weight.take(terminal_region, axis=0)

        #分子
        numerator = np.sum(sample_weight * residual)

        #分母
        denominator = np.sum(sample_weight * (y - residual) * (1 - y + residual))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf] = 0.0
        else:
            # *****************************************************
            # 这个是最关键的地方： 这个就是求解 delta:   F(m) = F(m-1) + learning_rate * delta
            #  delta = g/H   g 是 梯度， H是二次梯度..
            #  泰勒二次项展开, 求最优解，要求detal = g/H
            # https://web.stanford.edu/~hastie/Papers/AdditiveLogisticRegression/alr.pdf
            # http://statweb.stanford.edu/~jhf/ftp/trebst.pdf
            tree.value[leaf] = numerator / denominator  #why ?
            # *****************************************************







