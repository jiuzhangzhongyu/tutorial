# encoding=utf-8
"""
@author : zhongqing
"""

import  numpy as np
NULL = None
RAND_R_MAX = 100000000000
INFINITY = np.inf
FEATURE_THRESHOLD = 0.00001

class SplitRecord(object):
    def __init__(self):
        # 使用哪个feature 去split.
        self.feature = NULL
        self.pos = NULL
        self.threshold = NULL
        self.improvement = 0

        self.impurity_left = 0
        self.impurity_right = 0


class Splitter(object):
    def __init__(self, criterion, max_features, min_samples_leaf, random_state, presort):
        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        # self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.presort = presort
    def init(self, X , y, sample_weight):
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        n_samples = X.shape[0]

        self.n_samples = n_samples
        # self.samples = [0] * n_samples
        self.samples = range(n_samples)

        n_features = X.shape[1]

        self.features = range(n_features)

        self.n_features = n_features

        self.feature_values = [0] * n_samples
        self.constant_features = [0] * n_features

        self.y = y
        self.y_stride = 1 # todo  : buggy ?

        return 0
    def node_reset(self, start, end ):
        self.start = start
        self.end = end
        self.criterion.init(self.y, self.y_stride, self.n_samples, self.samples, start, end )
        return 0
    def node_split(self, impurity, split, n_constant_features):
        """
        Placeholder method
        :param impurity:
        :param split:
        :param n_constant_features:
        :return:
        """

        pass
    def node_value(self, dest):
        self.criterion.node_value(dest)
    def node_impurity(self):
        return self.criterion.node_impurity()

class BaseDenseSplitter(Splitter):
    def __init__(self,criterion, max_features, min_samples_leaf, min_weight_leaf, random_state, presort):
        self.X = NULL
        self.X_sample_stride = 0
        self.X_feature_stride = 0
        self.X_idx_sorted_ptr = NULL
        self.X_idx_sorted_stride = 0
        self.sample_mask = NULL
        self.presort = presort
    def init(self, X , y, sample_weight):
        super(BaseDenseSplitter, self).init(X, y )
        self.X = X
        self.X_sample_stride = 1
        self.X_feature_stride = 1


def _init_split(split_record , start_pos):
    self = split_record
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY

class BestSplitter(BaseDenseSplitter):
    def node_split(self, impurity, split, n_constant_features):
        """
        Find the best split on [start: end]

        这里针对 constant_feature做了一点优化处理.
        :param impurity:
        :param split:
        :param n_constant_features:
        :return:
        """
        samples = self.samples
        start= self.start
        end = self.end
        features = self.features
        constant_features = self.constant_features
        n_features = self.n_features
        X = self.X
        Xf = self.feature_values
        X_sample_stride = self.X_sample_stride
        X_feature_stride = self.X_feature_stride
        max_features = self.max_features
        min_samples_leaf = self.min_samples_leaf
        random_state =self.rand_r_state

        X_idx_sorted = self.X_idx_sorted_ptr

        sample_mask = self.sample_mask

        best = SplitRecord()
        current = SplitRecord()

        current_proxy_improvement = -np.inf
        best_proxy_improvement = -np.inf


        f_i = n_features
        f_j = 0
        n_visited_features = 0
        n_found_constants = 0
        n_drawn_constants = 0
        n_known_constants = n_constant_features[0]
        n_total_constants = n_known_constants


        current_feature_value = None
        partition_end = None

        _init_split(best, end)
        #   [n_drawn_constants, ... , n_known_constants,            f_i ]

        #  n_found_constants
        while f_i > n_total_constants and (  n_visited_features < max_features  or n_visited_features <= n_found_constants + n_drawn_constants):
            n_visited_features +=1

            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)

            if f_j < n_known_constants:
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp
                n_drawn_constants += 1
            else:
                f_j += n_found_constants

                current.feature = features[f_j]
                feature_offset = self.X_feature_stride * current.feature

                for i in range(start, end):
                    Xf[i] = X[samples[i] + feature_offset]


                sort(Xf + start , samples+ start, end-start)

                if Xf[end-1] <= Xf[start] + FEATURE_THRESHOLD: #cosntant
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature
                    n_found_constants +=1
                    n_total_constants +=1
                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j] , features[f_i]

                    self.criterion.reset()
                    p = start
                    while p < end:
                        while p+1 < end and Xf[p+1] <= Xf[p] + FEATURE_THRESHOLD: #skip constant
                            p+=1
                        p += 1

                        if p< end:
                            current.pos = p
                            if  (current.pos -start  < min_samples_leaf) or  (end - current.pos < min_samples_leaf):
                                continue
                            self.criterion.update(current.pos)

                            current_proxy_improvement = self.criterion.proxy_impurity_improvement()
                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                current.threshold = (Xf[p-1] + Xf[p])/2.0
                                if current.threshold  == Xf[p]:
                                    current.threshold = Xf[p-1]
                                best = current


        if best.pos  < end:
            feature_offset = X_feature_stride * best.feature
            partition_end = end
            p = start

            while p < partition_end :
                if X[X_sample_stride * samples[p] + feature_offset] <= best.threshold:
                    p +=1
                else:
                    partition_end = -1
                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp
            self.criterion.reset()
            self.criterion.update(best.pos)

            best.improvement = self.criterion.impurity_improvement(impurity)

            best.impurity_left, best.impurity_right = self.criterion.children_impurity()

        return best, n_total_constants





