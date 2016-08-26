import numpy as np
import pandas as pd
import json


class Base_Transfer(object):
    """
    Base class for transfer learning algorithms. Attributes should be
    initialized with outputs from parse_input.split_train_Test. See the
    documentation for that function for more information..

    Attributes:
        test_set_X: DataFrame representing feature matrix for test set.
        test_set_proj: Name of project sampled for test set.
        train_pool_X: DataFrame representing feature matrix for training set.
        train_pool_y: Series representing label vector for training set.
        train_pool_proj: Series the ith entry of which names the project to
            which the ith row of train_pool_X and train_pool_y belongs.
        Base_Classifier: The sklearn classifier used for all classification
            tasks. For example, sklearn.ensemble.RandomForestClassifier.
        rand_seed: Random seed passed to classifier  (default = 2016).
        classifier_params: Parameters passed to classifier (default = {})
    """

    def __init__(self, test_set_X, test_set_proj, train_pool_X, train_pool_y,
                 train_pool_proj, Base_Classifier, rand_seed=None,
                 classifier_params={}):
        """
        Instantiate transfer learning algorithm. See class documentation for
        more information about specific parameters. This function performs no
        operations beyond initializing class attributes.
        """

        self.test_set_X = test_set_X
        self.test_set_proj = test_set_proj
        self.train_pool_X = train_pool_X
        self.train_pool_y = train_pool_y
        self.train_pool_proj = train_pool_proj
        self.Base_Classifier = Base_Classifier
        self.rand_seed = rand_seed
        self.classifier_params = classifier_params

    def train_filter_test(self):

        raise NotImplementedError()

    def json_encode(self):
        """
        Encode transfer learning algorithm as JSON.

        Returns:
            Dictionary with keys 'Base_Classifier' and 'rand_seed', the base
            classifier and random seed passed to __init__. This dictionary can
            be passed to json.dumps to obtain a JSON formatted string.
        """

        return {
            "Base_Classifier": self.Base_Classifier,
            "rand_seed": self.rand_seed
        }
