import numpy as np
import pandas as pd
import json
from tl_algs import tl_alg


def sim_minmax(column):
    """Return a two-dimensional tuple where the first element is the minimum and the
    second is the max"""
    return min(column), max(column)


def sim_std(column):
    """Return a 2dim tuple of the form (mean - std, mean + std)"""
    return (np.mean(column) - np.std(column), np.mean(column) + np.std(column))


class GravityWeight(tl_alg.Base_Transfer):

    def __init__(
            self,
            test_set_X,
            test_set_proj,
            train_pool_X,
            train_pool_y,
            train_pool_proj,
            Base_Classifier,
            rand_seed=None,
            classifier_params={},
            similarity_func=sim_std):
        """Train a baseline classifier on only target data and
        return tuple (confidence, predictions).
        This classifier uses none of the source/Cross project data.
        """
        # this should be passing all the base parameters to the superclass
        super(
            GravityWeight,
            self).__init__(
            test_set_X,
            test_set_proj,
            train_pool_X,
            train_pool_y,
            train_pool_proj,
            Base_Classifier,
            rand_seed=rand_seed,
            classifier_params=classifier_params)
       # define the rest of the parameters here.
        self.similarity_func = similarity_func

    def train_filter_test(self):
        ranges = self.test_set_X.apply(self.similarity_func)

        def isinrange(row):
            return sum([cell >= ranges[i][0] and cell <= ranges[i][1]
                        for i, cell in enumerate(row)])
        # for each row, compute similarity
        similarity_count = self.train_pool_X.apply(isinrange, axis=1)
        weight_vec = map(lambda x: float(
            float(x) / (len(ranges) - x + 1)**2), similarity_count)
        # apply classification with sample weight.
        f = self.Base_Classifier(random_state=self.rand_seed,
                                 **self.classifier_params).fit(self.train_pool_X,
                                                               list(self.train_pool_y),
                                                               sample_weight=np.array(weight_vec))
        confidence = f.predict_proba(self.test_set_X)

        return ([a[-1] for a in confidence] if len(confidence[0]) >
                1 else list(confidence), f.predict(self.test_set_X))

    def json_encode(self):
        base = tl_alg.Base_Transfer.json_encode(self)
        base.update({"similarity_func": self.similarity_func.__name__})
        return base
