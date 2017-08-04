import numpy as np
import pandas as pd
import json
from tl_algs import tl_alg


class Target_Baseline(tl_alg.Base_Transfer):
    """
    Train classifier using only target or in-domain data, and no source or
    cross-domain data.
    """

    def __init__(self, test_set_X, test_set_domain, train_pool_X, train_pool_y,
                 train_pool_domain, Base_Classifier, rand_seed=None,
                 classifier_params={}):

        super(Target_Baseline, self).__init__(
            test_set_X,
            test_set_domain,
            train_pool_X,
            train_pool_y,
            train_pool_domain,
            Base_Classifier,
            rand_seed=rand_seed,
            classifier_params=classifier_params
        )

    def train_filter_test(self):
        """
        Train classifier using only target data and return class predictions
        and class-prediction probabilities. Note that the target baseline uses
        no source data.

        Returns:
            confidence: List of predicted-class probabilities, the ith entry
                of which gives the confidence value for the ith prediction.
            predictions: List of class predictions.
        """

        X_target = self.train_pool_X[
            np.array(self.train_pool_domain) == self.test_set_domain
        ]
        y_target = self.train_pool_y[
            np.array(self.train_pool_domain) == self.test_set_domain
        ]

        classifier = self.Base_Classifier(
            random_state=self.rand_seed,
            **self.classifier_params
        )

        f = classifier.fit(X_target, list(y_target))
        confidence = f.predict_proba(self.test_set_X)[:,-1]
        predictions = f.predict(self.test_set_X)

        return confidence, predictions

    def json_encode(self):

        return tl_alg.Base_Transfer.json_encode(self)


class Source_Baseline(tl_alg.Base_Transfer):
    """
    Train classifier using only source or cross-domain data, and no target or
    in-domain data.
    """

    def __init__(self, test_set_X, test_set_domain, train_pool_X, train_pool_y,
                 train_pool_domain, Base_Classifier, rand_seed=None,
                 classifier_params={}):

        super(Source_Baseline, self).__init__(
            test_set_X,
            test_set_domain,
            train_pool_X,
            train_pool_y,
            train_pool_domain,
            Base_Classifier,
            rand_seed=rand_seed,
            classifier_params=classifier_params
        )

    def train_filter_test(self):
        """
        Train classifier using only source data and return class predictions
        and class-prediction probabilities. Note that the source baseline uses
        no target data.

        Returns:
            confidence: List of predicted-class probabilities, the ith entry
                of which gives the confidence value for the ith prediction.
            predictions: List of class predictions.
        """

        X_source = self.train_pool_X[
            np.array(self.train_pool_domain) != self.test_set_domain
        ]
        y_source = self.train_pool_y[
            np.array(self.train_pool_domain) != self.test_set_domain
        ]

        classifier = self.Base_Classifier(
            random_state=self.rand_seed,
            **self.classifier_params
        )

        f = classifier.fit(X_source, list(y_source))
        confidence = f.predict_proba(self.test_set_X)
        predictions = f.predict(self.test_set_X)

        if len(confidence[0]) > 1:
            confidence = [a[-1] for a in confidence]
        else:
            confidence = list(confidence)

        return confidence, predictions

    def json_encode(self):

        return tl_alg.Base_Transfer.json_encode(self)


class Hybrid_Baseline(tl_alg.Base_Transfer):
    """
    Train classifier using all available source and target training data.
    """

    def __init__(self, test_set_X, test_set_domain, train_pool_X, train_pool_y,
                 train_pool_domain, Base_Classifier, rand_seed=None,
                 classifier_params={}):

        super(Hybrid_Baseline, self).__init__(
            test_set_X,
            test_set_domain,
            train_pool_X,
            train_pool_y,
            train_pool_domain,
            Base_Classifier,
            rand_seed=rand_seed,
            classifier_params=classifier_params
        )

    def train_filter_test(self):
        """
        Train classifier using all available data and return class predictions
        and class-prediction probabilities.

        Returns:
            confidence: List of predicted-class probabilities, the ith entry
                of which gives the confidence value for the ith prediction.
            predictions: List of class predictions.
        """

        classifier = self.Base_Classifier(
            random_state=self.rand_seed,
            **self.classifier_params
        )

        f = classifier.fit(self.train_pool_X, list(self.train_pool_y))
        confidence = f.predict_proba(self.test_set_X)
        predictions = f.predict(self.test_set_X)

        if len(confidence[0]) > 1:
            confidence = [a[-1] for a in confidence]
        else:
            confidence = list(confidence)

        return confidence, predictions

    def json_encode(self):

        return tl_alg.Base_Transfer.json_encode(self)
