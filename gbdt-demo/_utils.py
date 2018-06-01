# encoding=utf-8
"""
@author : zhongqing
"""


class StackRecord:
    def __init__(self, start, end ,depth, parent ,is_left, impurity, n_constant_features):
        self.start = start
        self.end = end
        self.depth =depth
        self.parent = parent
        self.is_left =is_left
        self.impurity = impurity
        self.n_constant_features = n_constant_features

class Stack(object):
    def __init__(self):

        self.buf = []
    def push(self, start, end , depth , parent, is_left, impurity, n_constant_features):
        a = StackRecord(
            start, end, depth, parent, is_left, impurity, n_constant_features
        )
        self.buf.append(a)

    def pop(self):

        return self.buf.pop()

    def is_empty(self):
        return len(self.buf) == 0

