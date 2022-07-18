import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from . import tl_alg, burak


class Peters(tl_alg.Base_Transfer):
    """
    Implements Peters' algorithm [1]. First, the nearest test instance from
    each training instance by Euclidean distance is computed. For each test
    instance, the closest training instance that selected that test instance in
    the previous step is retained. Finally, the classifier is trained on the
    filtered training set. Since computing the distance between every test
    instance and every training instance can be computationally expensive, this
    implementation follows [1] by clustering training instances using k-means,
    and applying the Peters filter within each cluster.

    [1] Fayola Peters, Tim Menzies, and Andrian Marcus. 2013. Better Cross-
    Company Defect Prediction. IEEE International Working Conference on Mining
    Software Repositories, 409-18.

    Attributes:
        cluster_factor: The ratio of instances to clusters for k-means. For
            example, a cluster factor of 100 yields approximately 100 instances
            per cluster.
    """

    def __init__(self, test_set_X, test_set_domain, train_pool_X, train_pool_y,
                 train_pool_domain, Base_Classifier, rand_seed=None,
                 classifier_params={}, cluster_factor=10):

        super(Peters, self).__init__(
            test_set_X,
            test_set_domain,
            train_pool_X,
            train_pool_y,
            train_pool_domain,
            Base_Classifier,
            rand_seed=rand_seed,
            classifier_params=classifier_params
        )

        self.cluster_factor = cluster_factor

    def filter_instances(self, train_pool_X, train_pool_y, X_test):
        """
        Implements the Peters filter. For each training instance r, find the
        nearest test instance by Euclidean distance. For each test instance e,
        let R(e) denote the set of training instances that selected e in the
        previous step. For each test instance e, the filtered training set
        contains the r in R(e) nearest to e.

        Args:
            train_pool_X: DataFrame representing training set feature matrix.
            train_pool_y: Series representing training set label vector.
            X_test: DataFrame representing test set feature matrix.

        Returns:
            train_X: Feature matrix for filtered training set.
            train_y: Label vector for filtered training set.
        """
        filter_X = pd.DataFrame()
        filter_y = []
        close_candidates = {}
        X_working = train_pool_X.reset_index(drop=True)
        y_working = train_pool_y.reset_index(drop=True)
        # finding the closest test instance to each training instance
        for train_index, row in X_working.iterrows():
            distances = euclidean_distances([row], X_test)[0]
            closest_index, closest_element = min(enumerate(distances), key=lambda x:x[1])
            # add the new closest training instance to each test instance
            close_candidates[closest_index] = (train_index, closest_element) \
                    if closest_index not in close_candidates.keys() \
                    else min((close_candidates[closest_index], (train_index, closest_element)),
                        key = lambda x: x[1])

        for index, __ in close_candidates.values():
            x_working = pd.DataFrame(X_working.loc[index,:]).transpose()
            filter_X = pd.concat([filter_X, x_working])
            filter_y.append(y_working.loc[index])

        return filter_X, pd.Series(filter_y)

    def peters_filter(self, test_set_X, test_set_domain, train_pool_X,
                      train_pool_y, train_pool_domain, Base_Classifier,
                      rand_seed=None, classifier_params={}):
        """
        Train classifier on filtered training data and return class predictions
        and predicted-class probabilities. See class documentation for more
        information on the form of this method's arguments.

        Returns:
            confidence: List of class-prediction probabilities, the ith entry
                of which gives the confidence for the ith prediction.
            prediction: List of class predictions.
        """

        X_filtered, y_filtered = filter_instances(
            train_pool_X,
            train_pool_y,
            test_set_X
        )

        classifier = Base_Classifier(
            random_state=rand_seed,
            **classifier_params
        )

        f = classifier.fit(X_filtered, y_filtered)
        confidence = [a[1] for a in f.predict_proba(test_set_X)]
        predictions = f.predict(test_set_X)

        return confidence, predictions

    def batch_peters_filter(self, test_set_X, test_set_domain, train_pool_X,
                            train_pool_y, train_pool_domain, Base_Classifier,
                            cluster_factor=10, rand_seed=None,
                            classifier_params={}):
        """
        Train classifier on filtered training data using the k-means heuristic
        and return class predictions and class-prediction probabilities. See
        class documentation for more information on the form of this method's
        arguments.

        Returns:
            confidence: List of class-prediction probabilities, the ith entry
                of which gives the confidence for the ith prediction.
            predictions: List of class predictions.
        """

        clusters = burak._kmeans_cluster(
            test_set_X,
            train_pool_X,
            train_pool_y,
            cluster_factor,
            rand_seed
        )

        X_train_filtered = pd.DataFrame()
        y_train_filtered = pd.Series(dtype=float)

        # Apply Peters filter within each cluster.
        for d in clusters:
            more_X_train, more_y_train = self.filter_instances(
                d['X_train'],
                d['y_train'],
                d['X_test']
            )
            # X_train_filtered = X_train_filtered.append(more_X_train)
            # y_train_filtered = y_train_filtered.append(more_y_train)
            X_train_filtered = pd.concat([X_train_filtered, more_X_train])
            y_train_filtered = pd.concat([y_train_filtered, more_y_train])
        X_train_filtered.reset_index(drop=True, inplace=True)
        y_train_filtered.reset_index(drop=True, inplace=True)

        classifier = Base_Classifier(
            random_state=rand_seed,
            **classifier_params
        )

        f = classifier.fit(X_train_filtered, y_train_filtered.tolist())
        confidence = [a[-1] for a in f.predict_proba(test_set_X)]
        predictions = f.predict(test_set_X)

        return confidence, predictions

    def train_filter_test(self):
        """
        Train classifier on filtered training data using the k-means heuristic
        and return class predictions and class-prediction probabilities. This
        method calls batch_peters_filter with class attributes.

        Returns:
            confidence: List of class-prediction probabilities, the ith entry
                of which gives the confidence for the ith prediction.
            predictions: List of class predictions.
        """

        return self.batch_peters_filter(
            self.test_set_X,
            self.test_set_domain,
            self.train_pool_X,
            self.train_pool_y,
            self.train_pool_domain,
            self.Base_Classifier,
            cluster_factor=self.cluster_factor,
            rand_seed=self.rand_seed,
            classifier_params=self.classifier_params
        )

    def json_encode(self):

        base = tl_alg.Base_Transfer.json_encode(self)
        base.update({"cluster_factor": self.cluster_factor})

        return base
