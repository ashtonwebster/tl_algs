import numpy as np
import pandas as pd
import json
from tl_algs import tl_alg
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

    master_X_df = train_pool_X.append(test_set_X)
    num_clust = master_X_df.shape[0] / cluster_factor

    kmeans = KMeans(n_clusters=num_clust, random_state=rand_seed)
    cluster_model = kmeans.fit(master_X_df)

    clusters = [{
        'X_train': pd.DataFrame(),
        'y_train': pd.Series(),
        'X_test': pd.DataFrame(),
        'y_test': pd.Series()
    } for i in range(num_clust)]

    # Populate clusters based on test data.
    X_test_clusters = cluster_model.predict(test_set_X)
    for i, clust in enumerate(X_test_clusters):
        clusters[clust]['X_test'] = clusters[clust]['X_test']. \
            append(test_set_X.iloc[i, ])

    # Populate clusters based on training data.
    X_train_clusters = cluster_model.predict(train_pool_X)
    for i, clust in enumerate(X_train_clusters):
        clusters[clust]['X_train'] = clusters[clust]['X_train']. \
            append(train_pool_X.iloc[i, ])
        clusters[clust]['y_train'] = clusters[clust]['y_train'] \
            .append(pd.Series([train_pool_y.iloc[i]]))

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

        # Copy train_pool_X so that original is not modified.
        train_pool_X_local = train_pool_X.copy()
        train_pool_y = list(train_pool_y)
        output_train_X = pd.DataFrame(columns=train_pool_X.columns)
        output_train_y = []

        # For each test instance, find distance to each training instance.
        dist_matrix = euclidean_distances(X_test, train_pool_X_local)

        # For each distance, append reference to original X_train_pool row and
        # label. This gives a list of  the form [[{euc_dist, x, label}]].
        dist_matrix = [[{
            "euc_dist": dist_matrix[i][j],
            "x": train_pool_X_local.iloc[j, :],
            "y": train_pool_y[j]
        } for j in range(len(dist_matrix[0]))
        ] for i in range(len(dist_matrix))]

        # Indices of training instances that have not been selected.
        available_training_ind = range(len(dist_matrix[0]))
        for i, x_test_instance in enumerate(X_test.iterrows()):
            # Sort training instances by distance from current test instance.
            available_training_ind = sorted(
                available_training_ind,
                key=lambda a: dist_matrix[i][a]['euc_dist']
            )

            # Take the top k training instances.
            for j, x_train_instance in enumerate(
                    [dist_matrix[i][a] for a in available_training_ind][:k]):
                # Append each instance to output.
                if x_train_instance['x'].name not in output_train_X.index:
                    output_train_X = output_train_X.append(x_train_instance['x'])
                    output_train_y.append(x_train_instance['y'])

            # Remove training instances from list to ensure uniqueness.
            #del available_training_ind[:k]

        return output_train_X.drop_duplicates().reset_index(drop=True), pd.Series(output_train_y)

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
        y_train_filtered = pd.Series()

        # Apply Burak filter within each cluster.
        for d in clusters:
            more_X_train, more_y_train = self.filter_instances(
                d['X_train'],
                d['y_train'],
                d['X_test'],
                k
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
