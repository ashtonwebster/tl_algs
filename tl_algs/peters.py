import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from tl_algs import tl_alg, burak


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

        output_X = pd.DataFrame(columns=train_pool_X.columns)
        output_y = []

        # For each training instance, find distance to each test instance.
        dist_matrix = euclidean_distances(train_pool_X, X_test)

        # For each distance, append reference to original X_train_pool row and
        # label. This gives a list of the form [[{euc_dist_to, x, y, label}]].
        dist_matrix = [[{
            "euc_dist_to": None,
            "euc_dist_from": dist_matrix[i][j],
            "x": train_pool_X.iloc[i, :],
            "y": train_pool_y.iloc[i]
        } for j in range(len(dist_matrix[0]))
        ] for i in range(len(dist_matrix))]

        # For each index i corresponding to test instance e_i, there exists a
        # set of training instances R_i such that e_i is the closest test
        # instance in e_i to ever training instance r in R_i.
        fans = [[] for i in range(len(X_test))]

        # Get closest test instance for each training instance.
        for i, (name, X_train_instance) in enumerate(train_pool_X.iterrows()):
            # Sort by closest test instance to each training instance.
            closest_test_name = np.argmin(
                [a['euc_dist_from'] for a in dist_matrix[i]]
            )

            # Append information about each instance to each fan.
            fans[closest_test_name].append(dist_matrix[i][closest_test_name])

            # For each new fan, find the largest fan in each group (i.e., the
            # training instance closest to the test instance) and retain it.
            for i, fan_group in enumerate(fans):
                if len(fan_group) > 0:
                    test_instance = [X_test.iloc[i, :]]
                    fan_group_distances = [
                        a['euc_dist_from'] for a in fan_group
                    ]
                    min_index = np.argmin(fan_group_distances)
                    output_X = output_X.append(fan_group[min_index]['x'])
                    output_y.append(fan_group[min_index]['y'])

        return output_X, pd.Series(output_y)

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
        y_train_filtered = pd.Series()

        # Apply Peters filter within each cluster.
        for d in clusters:
            more_X_train, more_y_train = self.filter_instances(
                d['X_train'],
                d['y_train'],
                d['X_test']
            )
            X_train_filtered = X_train_filtered.append(more_X_train)
            y_train_filtered = y_train_filtered.append(more_y_train)

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
