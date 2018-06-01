# encoding=utf-8
"""
@author : zhongqing
"""

TREE_LEAF = -1
def predict_stages(estimators, X , scale, score):
    """Add predictions of ``estimators`` to ``out``.
   Each estimator is scaled by ``scale`` before its prediction
   is added to ``out``.
   """
    n_estimators = estimators.shape[0]
    for i in range(n_estimators):
        tree = estimators[i].tree_
        _predict_regression_tree_inplace_fast_dense(X, tree.nodes, tree.value, scale, X.shape[0], X.shape[1], score)


def predict_stage(estimators, stage, X, scale, score):
    return predict_stages(estimators[stage:stage+1], X, scale,score)



def _predict_regression_tree_inplace_fast_dense(X, root_node, value, scale, n_samples, n_features, score):
    """Predicts output for regression tree and stores it in ``out[i, k]``.

     This function operates directly on the data arrays of the tree
    data structures. This is 5x faster than the variant above because
    it allows us to avoid buffer validation.
    The function assumes that the ndarray that wraps ``X`` is
    c-continuous.

    一种简单的优化实现.
    """
    for i in range(n_samples):
        node = root_node
        while node.left_child != TREE_LEAF:
            if X [ i *  n_features + node.feature] <= node.threshold:
                node = root_node + node.left_child
            else:
                node = root_node + node.right_child

        score[i]  = scale * value[node-root_node]


