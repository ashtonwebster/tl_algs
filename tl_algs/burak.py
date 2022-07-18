import numpy as np
import pandas as pd
import json
from . import tl_alg
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def _kmeans_cluster(test_set_X, train_pool_X, train_pool_y, cluster_factor,
                    rand_seed):
    """
    Partition dataset into clusters using k-means. The number of clusters is
    the number of training and test instances divided by the cluster factor.

    Args:
        test_set_X: DataFrame representing feature matrix for test set.
        train_pool_X: DataFrame respresenting feature matrix for training set.
        train_pool_y: Series representing label vector for training set.
        cluster_factor: Ratio of instances to clusters. For example, a cluster
            factor of 100 yields approximately 100 instances per cluster.
        rand_seed: Random seed passed to sklearn.cluster.KMeans.

    Returns:
        List of dictionaries with keys 'X_train', a DataFrame of training
        instances; 'y_train', a Series of labels for X_train; and 'X_test', a
        DataFrame of test instances. This list has the property that its ith
        entry is the ith cluster of training and test instances.
    """

    master_X_df = pd.concat([train_pool_X, test_set_X])
    num_clust = master_X_df.shape[0] // cluster_factor

    kmeans = KMeans(n_clusters=num_clust, random_state=rand_seed)
    cluster_model = kmeans.fit(master_X_df)

    clusters = [{
        'X_train': pd.DataFrame(),
        'y_train': pd.Series(dtype=float),
        'X_test': pd.DataFrame(),
        'y_test': pd.Series(dtype=float)
    } for i in range(num_clust)]

    # Populate clusters based on test data.
    X_test_clusters = cluster_model.predict(test_set_X)
    for i, clust in enumerate(X_test_clusters):
        x_pool = pd.DataFrame(test_set_X.iloc[i, ]).transpose()
        clusters[clust]['X_test'] = pd.concat([clusters[clust]['X_test'], x_pool])

    # Populate clusters based on training data.
    X_train_clusters = cluster_model.predict(train_pool_X)
    for i, clust in enumerate(X_train_clusters):
        x_pool = pd.DataFrame(train_pool_X.iloc[i, ]).transpose()
        y_pool = pd.Series([train_pool_y.iloc[i]])
        clusters[clust]['X_train'] = pd.concat([clusters[clust]['X_train'], x_pool])
        clusters[clust]['y_train'] = pd.concat([clusters[clust]['y_train'], y_pool])

    # Remove clusters with no test instance.
    to_remove = [
        i for (i, d) in enumerate(clusters)
        if (d['X_test'].shape[0] == 0 or d['X_train'].shape[0] == 0)
    ]

    to_remove.reverse()
    for i in to_remove:
        del clusters[i]

    return clusters


class Burak(tl_alg.Base_Transfer):
    """
    Implements Burak's algorithm [1]. First, the training set is filtered using
    k-nearest neighbors, so that for each test instance, the k unique nearest
    training instances by Euclidean distance are retained. Next, the classifier
    is trained on the filtered training instances. Since computing the distance
    between every test instance and every training instance is computationally
    expensive, this implementation clusters training instances using k-means
    and applies Burak's algorithm within each cluster [2].

    [1] Burak Turhan, Tim Menzies, Ayse B. Bener, and Justin Di Stefano. 2009.
    "On the Relative Value of Cross-Company and Within-Company Data for Defect
    Prediction." Empirical Software Engineering 14, 5, 540-78.

    [2] Fayola Peters, Tim Menzies, and Andrian Marcus. 2013. "Better Cross-
    Company Defect Prediction. IEEE International Working Conference on Mining
    Software Repositories, 409-18.

    Attributes:
        k: The number of unique nearest training instances to retain for each
            target test instance. Following [1], the default value is 10.
        cluster_factor: The ratio of instances to clusters for k-means. For
            example, a cluster factor of 100 yields approximately 100 instances
            per cluster.
    """

    def __init__(self, test_set_X, test_set_domain, train_pool_X, train_pool_y,
                 train_pool_domain, Base_Classifier, classifier_params={},
                 rand_seed=None, k=10, cluster_factor=100):

        super(Burak, self).__init__(
            test_set_X,
            test_set_domain,
            train_pool_X,
            train_pool_y,
            train_pool_domain,
            Base_Classifier,
            rand_seed=rand_seed,
            classifier_params=classifier_params
        )

        self.k = k
        self.cluster_factor = cluster_factor

    def filter_instances(self, train_pool_X, train_pool_y, X_test, k):
        """
        For each test instance, retain the k unique nearest training instances
        by Euclidean distance.

        Args:
            train_pool_X: DataFrame representing training set feature matrix.
            train_pool_y: Series representing training set label vector.
            X_test: DataFrame representing test set feature matrix.
            k: For each test instance, the number of unique nearest training
                instances to retain.

        Returns:
            train_X: Feature matrix for filtered training set.
            train_y: Label vector for filtered training set.
        """
        filtered_X = pd.DataFrame()
        filtered_y = []
        working_X = train_pool_X.reset_index(drop=True)
        working_y = list(train_pool_y)
        # for each instance in the the test set
        for (__, row) in X_test.iterrows():
            # find distances to all instances in training pool
            distances = euclidean_distances([row], train_pool_X)[0]
            # get indexes of closest instances
            sorted_distance_indexes = [index for index, __ in sorted(enumerate(distances), key=lambda x:x[1])]
            # Add top k closest instances to output
            for i in sorted_distance_indexes[:k]:
                if i not in filtered_X.index:
                    x_working = pd.DataFrame(working_X.iloc[i,:]).transpose()
                    filtered_X = pd.concat([filtered_X, x_working])
                    filtered_y.append(working_y[i])

        return filtered_X, pd.Series(filtered_y)

    def burak_filter(self, test_set_X, test_set_domain, train_pool_X,
                     train_pool_y, train_pool_domain, Base_Classifier, k=10,
                     rand_seed=None, classifier_params={}):
        """
        Train classifier on filtered training data and return class predictions
        and predicted-class probabilities. See class documentation for more
        information on the form of this method's arguments.

        Returns:
            confidence: List of class-prediction probabilities, the ith entry
                of which gives the confidence for the ith prediction.
            predictions: List of class predictions.
        """
        X_filtered, y_filtered = self.filter_instances(
            train_pool_X,
            train_pool_y,
            test_set_X,
            k
        )

        classifier = Base_Classifier(
            random_state=rand_seed,
            **classifier_params
        )

        f = classifier.fit(X_filtered, y_filtered)
        confidence = [a[1] for a in f.predict_proba(test_set_X)]
        predictions = f.predict(test_set_X)

        return confidence, predictions

    def batch_burak_filter(self, test_set_X, test_set_domain, train_pool_X,
                           train_pool_y, train_pool_domain, Base_Classifier,
                           k=10, rand_seed=None, cluster_factor=100,
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

        clusters = _kmeans_cluster(
            test_set_X,
            train_pool_X,
            train_pool_y,
            cluster_factor,
            rand_seed
        )

        X_train_filtered = pd.DataFrame()
        y_train_filtered = pd.Series(dtype=float)

        # Apply Burak filter within each cluster.
        for d in clusters:
            more_X_train, more_y_train = self.filter_instances(
                d['X_train'],
                d['y_train'],
                d['X_test'],
                k
            )
            X_train_filtered = pd.concat([X_train_filtered, more_X_train])
            y_train_filtered = pd.concat([y_train_filtered, more_y_train])

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
        method calls batch_burak_filter with class attributes.

        Returns:
            confidence: List of class-prediction probabilities, the ith entry
                of which gives the confidence for the ith prediction.
            predictions: List of class predictions.
        """

        return self.batch_burak_filter(
            self.test_set_X,
            self.test_set_domain,
            self.train_pool_X,
            self.train_pool_y,
            self.train_pool_domain,
            self.Base_Classifier,
            k=self.k,
            rand_seed=self.rand_seed,
            cluster_factor=self.cluster_factor,
            classifier_params=self.classifier_params
        )

    def json_encode(self):

        base = tl_alg.Base_Transfer.json_encode(self)
        base.update({"cluster_factor": self.cluster_factor})

        return base
