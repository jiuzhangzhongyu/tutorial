# encoding=utf-8
"""
@author : zhongqing
"""
import numpy as np
from utils import StackRecord, Stack


_TREE_UNDEFINED = -1
_TREE_LEAF = -2
EPSILON = 0.001
class TreeBuilder(object):
    def __init__(self):
        pass
    def build(self, tree, X , y ):
        pass
class DepthFirstTreeBuilder(TreeBuilder):
    def __init__(self, splitter, min_samples_split, min_samples_leaf,max_depth, min_impurity_decrease, min_impurity_split):
        self.splitter=  splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        pass
    def build(self, tree, X , y):

        init_capacity =  0
        if tree.max_depth <= 10:
            init_capacity = ( 2** (tree.max_depth+1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        splitter = self.splitter
        max_depth = self.max_depth
        min_samples_leaf = self.min_samples_leaf
        min_samples_split = self.min_samples_split
        min_impurity_decrease = self.min_impurity_decrease
        min_impurity_split = self.min_impurity_split

        splitter.init(X, y)
        n_node_samples = splitter.n_samples

        # SplitRecord split
        impurity = np.inf
        first = 1
        max_depth_seen = -1
        rc = 0
        stack = Stack()
        # stack_record

        # node = StackRecord()
        # node.start = 0
        # node.end = n_node_samples
        # node.depth = 0
        # node.parent = _TREE_UNDEFINED
        # node.is_left = 0
        # node.impurity = np.inf
        # node.n_constant_features = 0

        stack.push(0, n_node_samples, 0 , _TREE_UNDEFINED, 0,np.inf, 0 )
        while  not stack.is_empty() > 0:
            node = stack.pop()

            start = node.start
            end = node.end
            is_left = node.is_left
            is_leaf = node.is_leaf
            depth = node.depth

            n_node_samples = node.end - node.start
            splitter.node_reset(node.start, node.end)

            is_leaf = (node.depth >= max_depth or n_node_samples < min_samples_split or n_node_samples < 2* min_samples_leaf )

            if first:
                impurity = splitter.node_impurity()
                first = 0
            is_leaf = (is_leaf or (impurity <= min_impurity_split))

            if not is_leaf:
                split , n_constant_features = splitter.node_split(impurity)
                is_leaf = (is_leaf or split.pos >= node.end or (split.improvement + EPSILON < min_impurity_decrease))

            node_id = tree._add_node(node.parent, node.is_left, is_leaf, split.feature, split.threshold, node.impurity, n_node_samples)

            splitter.node_value(tree.value + node_id * tree.value_stride)

            if not is_leaf:
                stack.push(split.pos,end , depth + 1, node_id, 0, split.impurity_right, n_constant_features)
                stack.push(start,split.pos, depth+1, node_id, 1, split.impurity_left, n_constant_features)
            if depth > max_depth_seen:
                max_depth_seen = depth

            tree.max_depth = max_depth_seen








class Node(object):
    def __init__(self):
        self.node_id = 0
        self.left_child = 0
        self.right_child = 0
        self.feature = 0
        self.threshold = 0
        self.impurity = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0


class Tree(object):
    """
     # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.

     Array-based representation of a binary decision tree.
    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    ~~:  baipeng
    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count]
        Contains the constant prediction value of each node.

        Following deprecated: 目前只针对二分类问题.
        value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.


    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.


    """
    def __init__(self):
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.nodes = []
        for i in range(self.capacity):
            self.nodes.append(Node())

        self.value = []

        self.value_stride = 1.
        self.max_n_classes = 1
        self.n_outputs = 1

    def _add_node(self, parent, is_left , is_leaf, feature, threshold, impurity, n_node_samples,weighted_n_node_samples):
        """Add a node to the tree.
               The new node registers itself as the child of its parent.
               Returns (size_t)(-1) on error.
               """
        node_id = self.node_count
        node = self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples
        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED
        else:
            node.feature = feature
            node.threshold = threshold

        self.node_count +=1
        return node_id
    def _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.
               The array keeps a reference to this Tree, which manages the underlying
               memory.
               """
        # node_count , n_outputs , max_n_classes .
        raise NotImplementedError("Not implemented")

    def _get_node_ndarray(self):
        raise NotImplementedError("Not implemented")


    def apply(self, X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        return self._apply_dense(X)
    def _apply_dense(self, X):
        n_samples = X.shape[0]
        out = np.zeros((n_samples, ))
        for i  in range(n_samples):
            node = self.nodes
            while node.left_child != _TREE_LEAF:
                if X[i][node.feature] <= node.threshold:
                    node = self.nodes[node.left_child]
                else:
                    node = self.nodes[node.right_child]
            out[i] = node.node_id

        return out

    def predict(self, X ):

        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        return out.ravel()

    def compute_feature_importances(self,normalize=True):
        """Computes the importance of each feature (aka variable)."""

        importances = np.zeros((self.n_features, ))
        nodes = self.nodes
        i = 0
        while i < self.node_count:
            node = self.nodes[i]

            if node.left_child != _TREE_LEAF:
                left =  nodes[node.left_child]
                right = nodes[node.right_child]

                importances[node.feature] += (
                    node.weighted_n_node_samples * node.impurity -
                    left.weighted_n_node_samples * left.impurity -
                    right.weighted_n_node_samples * right.impurity
                )
            i +=1

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)
            if normalizer > 0.0:
                importances /= normalizer

        return importances

    @property
    def children_left(self):
        return self._get_node_ndarray()['left_child'][:self.node_count]

    @property
    def children_left(self):
        return self._get_node_ndarray()['right_child'][:self.node_count]

    @property
    def feature(self):
        return self._get_node_ndarray()['feature'][:self.node_count]

    @property
    def threshold(self):
        return self._get_value_ndarray()['threshold'][:self.node_count]

    @property
    def impurity(self):
        return self._get_value_ndarray()['impurity'][:self.node_count]

    @property
    def n_node_samples(self):
        return self._get_value_ndarray()['n_node_samples'][:self.node_count]

    @property
    def weighted_n_node_samples(self):
        return self._get_value_ndarray()['weighted_n_node_samples'][:self.node_count]

    @property
    def value(self):
        return self._get_value_ndarray()[:self.node_count]

















