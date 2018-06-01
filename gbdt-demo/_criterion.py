# encoding=utf-8
"""
@author : zhongqing
"""

import numpy as np

NULL = None

class Criterion:
    def __init__(self):
        pass
    def init(self, y, y_stride, sample_weight,weighted_n_samples, samples, start, end  ):
        """

        :param y:
        :param y_stride:
        :param sample_weight:
        :param weighted_n_samples:
        :param samples:
        :param start:
        :param end:
        :return:
        """
        pass
    def reset(self):
        pass
    def reverse_reset(self):
        pass

    def update(self, new_pos):
        pass
    def node_impurity(self):
        pass
    def children_impurity(self):
        """

        :return: left_impurity , right_impurity
        """
        pass
    def node_value(self):
        """
        Placeholder for storing the node value.
        :return:
        """
        pass
    def proxy_impurity_improvement(self):
        """
        :return:
        """
        impurity_left,  impurity_right = self.children_impurity()
        return  - impurity_right - impurity_left

    def impurity_improvement(self, impurity):
        """
        按照比例过滤.
        :param impurity:
        :return:
        """
        impurity_left, impurity_right = self.children_impurity()

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - (self.weighted_n_right /
                             self.weighted_n_node_samples * impurity_right)
                 - (self.weighted_n_left /
                    self.weighted_n_node_samples * impurity_left)))


class RegressionCriterion(Criterion):
    """
    Abstract regression criterion
    var = \sum_i^n (y_i - y_bar) ** 2
    """
    def __init__(self, n_outputs, n_samples):

        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL
    def init(self, y, y_stride, sample_weight, weighted_n_samples, samples, start, end):
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.
        self.sq_sum_total = 0.0
        ## weighted 相关内容.
        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = y[i * y_stride + k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
    def reset(self):
        """
        全分在右边
        :return:
        """
        self.sum_left = 0
        self.sum_right = self.sum_total
        self.weighted_n_left = 0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0
    def reverse_reset(self):
        """
        全分在左边
        :return:
        """
        self.sum_right = 0
        self.sum_left = self.sum_total
        self.weighted_n_right = 0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end


    def node_impurity(self):
        pass
    def children_impurity(self):
        pass

    def node_value(self):
        """
        这里其实就是最终结算出来的 impurity
        :return:
        """
        return self.sum_total/self.weighted_n_node_samples
    def update(self, new_pos):
        """Updated statistics by moving samples[pos:new_pos] to the left."""
        sum_left =self.sum_left
        sum_right = self.sum_right
        sum_total = self.sum_total

        sample_weight = self.sample_weight

        samples = self.samples

        y = self.y
        pos= self.pos
        end = self.end
        w = 1.0

        if (new_pos - pos ) <= ( end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                y_ik = y[i * self.y_stride ]
                sum_left += w * y_ik

                self.weighted_n_left += w
        else:
            self.reverse_reset()
            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]
                y_ik = y[i*self.y_stride ]
                sum_left -= w * y_ik
                self.weighted_n_left -= w

        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        self.sum_right = sum_total - sum_left
        self.pos = new_pos

        return 0



class MSE(RegressionCriterion):
    """
    MSE = var_left + var_right
    """
    def node_impurity(self):
        """
        这里用到了一个简单的方差公式变换.
        :return:
        """
        sum_total = self.sum_total

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        impurity -= (sum_total / self.weighted_n_node_samples)**2

        return impurity

    def proxy_impurity_improvement(self):

        proxy_impurity_left = self.sum_left * self.sum_left


        proxy_impurity_right = self.sum_right * self.sum_right

        return proxy_impurity_left/self.weighted_n_left + proxy_impurity_right/self.weighted_n_right

    def children_impurity(self):
        start = self.start
        pos = self.pos
        y = self.y  #nd.array
        samples = self.samples

        sum_left = self.sum_left
        sum_right = self.sum_right

        sq_sum_left = 0
        for p in range(start, pos):
            i = samples[p]

            sq_sum_left += y[i * self.y_stride ] * y[i * self.y_stride ]

        sq_sum_right = self.sq_sum_total - sq_sum_left


        impurity_left =  sq_sum_left/self.weighted_n_left  - (sum_left/self.weighted_n_left) ** 2.0
        impurity_right = sq_sum_right/self.weighted_n_right - (sum_right/self.weighted_n_right) ** 2.0

        return impurity_left, impurity_right





