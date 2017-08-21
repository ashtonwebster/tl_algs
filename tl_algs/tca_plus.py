import numpy as np
import pandas as pd
import json
from tl_algs import tl_alg
import da_tool.tca
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import zscore
from enum import Enum

class SimLevel(Enum):
    MUCH_LESS = 0
    LESS = 1
    SLIGHTLY_LESS = 2
    SAME = 3
    SLIGHTLY_MORE = 4
    MORE = 5
    MUCH_MORE = 6

class TCAPlus(tl_alg.Base_Transfer):
    """
    A normalization based method of transfer learning based on [1]

    [1] Nam, J., Pan, S. J., & Kim, S. (2013). Transfer defect learning. 
    In Proceedings - International Conference on Software Engineering 
    (pp. 382-391). https://doi.org/10.1109/ICSE.2013.6606584
    """

    def __init__(self, test_set_X, test_set_domain, train_pool_X, train_pool_y,
                 train_pool_domain, Base_Classifier, classifier_params={},
                 rand_seed=None, dims=5, kernel_type='linear', kernel_param=1,
                mu=1):

        super(TCAPlus, self).__init__(
            test_set_X,
            test_set_domain,
            train_pool_X,
            train_pool_y,
            train_pool_domain,
            Base_Classifier,
            rand_seed=rand_seed,
            classifier_params=classifier_params
        )

        self.dims = dims
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.mu = mu

    def no_normalization(self, source_X, target_X):
        """ Corresponds to NoN in the paper [1] """
        return source_X, target_X

    def minmax_normalization(self, X_source, X_target):
        """ Corresponds to N1 in the paper [1] """
        all_X = X_source.append(X_target)
        X_source_norm = X_source.copy()
        X_target_norm = X_target.copy()
        for feature_index in range(all_X.shape[1]):
            min_ = all_X.iloc[:, feature_index].min()
            max_ = all_X.iloc[:, feature_index].max()
            range_ = max_ - min_
            X_source_norm.iloc[:, feature_index] = \
                    (X_source_norm.iloc[:, feature_index] - min_) / range_ \
                    if range_ != 0 else 0
            X_target_norm.iloc[:, feature_index] = \
                    (X_target_norm.iloc[:, feature_index] - min_) / range_ \
                    if range_ != 0 else 0
        return X_source_norm, X_target_norm

    def zscore_normalization(self, X_source, X_target):
        """ Corresponds to N2 in the paper [1]
        This provides normalization by using zscore on both source
        and target data"""
        all_X = X_source.append(X_target)
        X_source_norm = X_source.copy()
        X_target_norm = X_target.copy()
        for feature_index in range(all_X.shape[1]):
            mean_ = np.mean(all_X.iloc[:, feature_index])
            std_ = np.std(all_X.iloc[:, feature_index])
            X_source_norm.iloc[:, feature_index] = \
                    (X_source_norm.iloc[:, feature_index] - mean_) / std_ \
                    if std_ != 0 else 0
            X_target_norm.iloc[:, feature_index] = \
                    (X_target_norm.iloc[:, feature_index] - mean_) / std_ \
                    if std_ != 0 else 0
        return X_source_norm, X_target_norm

    def zscore_source_normalization(self, X_source, X_target):
        """ Corresponds to N3 in the paper [1] """
        X_source_norm = X_source.copy()
        X_target_norm = X_target.copy()
        for feature_index in range(X_source.shape[1]):
            mean_ = np.mean(X_source.iloc[:, feature_index])
            std_ = np.std(X_source.iloc[:, feature_index])
            X_source_norm.iloc[:, feature_index] = \
                    (X_source_norm.iloc[:, feature_index] - mean_) / std_ \
                    if std_ != 0 else 0
            X_target_norm.iloc[:, feature_index] = \
                    (X_target_norm.iloc[:, feature_index] - mean_) / std_ \
                    if std_ != 0 else 0
        return X_source_norm, X_target_norm

    def zscore_target_normalization(self, X_source, X_target):
        """Corresponds to N4 in the paper [1] """
        X_source_norm = X_source.copy()
        X_target_norm = X_target.copy()
        for feature_index in range(X_target.shape[1]):
            mean_ = np.mean(X_target.iloc[:, feature_index])
            std_ = np.std(X_target.iloc[:, feature_index])
            X_source_norm.iloc[:, feature_index] = \
                    (X_source_norm.iloc[:, feature_index] - mean_) / std_ \
                    if std_ != 0 else 0
            X_target_norm.iloc[:, feature_index] = \
                    (X_target_norm.iloc[:, feature_index] - mean_) / std_ \
                    if std_ != 0 else 0
            return X_source_norm, X_target_norm

    def compute_distance_set(self, proj_X):
        #proj_X = self.train_pool_X[np.array(self.train_pool_domain) \
        #        == project_name]
        dist_full = euclidean_distances(proj_X, proj_X)
        # there are duplicates in dist full (It is symmetric
        # about the diagonal) so we get the unique values
        dist = []
        for i in range(dist_full.shape[0]):
            for j in range(i):
                dist.append(dist_full[i,j])
        return dist

    def compute_dcv(self, dist, count):
        """ Computes a distance characteristic vector (DCV)
        from the distance set DIST 
        
        Args:
            dist: the distance set
            count: the original number of instances in the set
        
        Returns:
            A dict with keys "mean", "median", "min", "max", "std",
            and "len"
        """
        return { "mean" : np.mean(dist), "median" : np.median(dist),
                "min" : np.min(dist), "max" : np.max(dist),
                "std" : np.std(dist), "len" : count }

    def compute_comp_similarity(self, c_s, c_t):
        """ Computes the nominal degree of similarity between
        the source distance component (c_S) and the target distance
        component (c_t)  
        
        Args:
            c_s: source component of distance vector
            c_t: target component of distance vector
            
        Returns:
            SimLevel
        """
        ranges = [-9999, 0.4, 0.7, 0.9, 1.1, 1.3, 1.6, 99999]
        for i in range(1, len(ranges)):
            left_edge, right_edge = ranges[i-1], ranges[i]
            if c_s * left_edge <= c_t and c_t < c_s * right_edge:
                return SimLevel(i-1)
        Exception("No similarity level found")

    def compute_dist_similarity(self, dcv_s, dcv_t):
        """ 
        Computes the similarity for each component of the DCV vectors for the
        source and target projects 
        
        Args:
            dcv_s : Distance characteristic vector for source project
            dcv_t :  Distance characteristic vector for target project

        Returns:
            Dictionary with same keys with values as the component similarities
        """
        return { key : self.compute_comp_similarity(dcv_s[key], dcv_t[key]) for \
                key in dcv_s.keys() }

    def apply_normalization(self, sim_vector, source_X, target_X):
        """
        Applies normalization according to the rules specified in
        [1]

        Args:
            dist_s: the source distance set
            dist_t: the target distance set
        
        Returns:
            Normalized project data
        """
        # Rule 1
        if ( sim_vector['mean'] == SimLevel.SAME) \
            and ( sim_vector['std'] == SimLevel.SAME):
            return self.no_normalization(source_X, target_X)
        # Rule 2
        if all(sim_vector[key] in (SimLevel.MUCH_LESS, SimLevel.MUCH_MORE) \
                for key in ['len', 'min', 'max']):
            return self.minmax_normalization(source_X, target_X)
        # Rule 3
        if (sim_vector['std'] == SimLevel.MUCH_MORE \
                and sim_vector['len'].value < SimLevel.SAME.value) or \
           (sim_vector['std'] == SimLevel.MUCH_LESS \
                and sim_vector['len'].value > SimLevel.SAME.value):
               return self.zscore_source_normalization(source_X, target_X)
        # Rule 4
        if (sim_vector['std'] == SimLevel.MUCH_MORE \
                and sim_vector['len'] == SimLevel.MUCH_MORE) or \
           (sim_vector['std'] == SimLevel.MUCH_LESS \
                and sim_vector['len'] == SimLevel.MUCH_LESS):
               return  self.zscore_target_normalization(source_X, target_X)
        # Rule 5
        return self.zscore_normalization(source_X, target_X)

    def predict_one_proj(self, source_X, source_y, target_X, test_dist, test_dcv):
        # compute dcv for project
        X_dist = self.compute_distance_set(source_X)
        X_dcv = self.compute_dcv(X_dist, source_X.shape[0])
        # compute similarity vector"
        s = self.compute_dist_similarity(X_dcv, test_dcv)
        # normalize according to rules
        source_X_norm, target_X_norm = \
                self.apply_normalization(s, source_X, target_X)
        # apply TCA transformation 
        my_tca = da_tool.tca.TCA(dim=self.dims,kerneltype=self.kernel_type, 
                kernelparam=self.kernel_param, mu=self.mu)
        source_TCA_X, target_TCA_X, __ = \
            my_tca.fit_transform(source_X_norm, target_X_norm)
        target_TCA_X = pd.DataFrame(target_TCA_X, index=target_X_norm.index)
        # separating out original test set instances
        test_TCA_X = target_TCA_X.iloc[:self.test_set_X.shape[0], :]
        # train classifier and collect results for project classifier
        clf = self.Base_Classifier(random_state=self.rand_seed,
                **self.classifier_params)
        clf.fit(source_TCA_X, source_y)
        return clf.predict(test_TCA_X)


    def train_filter_test(self):
        """
        Train according to implementation described in paper [1].
        
        Returns:
            confidence: List of class-prediction probabilities, the ith entry
                of which gives the confidence for the ith prediction.
            predictions: List of class predictions.
        """
        # compute dcv for target data (include train and test data)
        target_train = self.train_pool_X[np.array(self.train_pool_domain) \
                == self.test_set_domain]
        # note that first n instances are test set, these are later
        # retrieved and separated from training target instances before
        # prediction
        target_X = self.test_set_X.append(target_train)
        test_dist = self.compute_distance_set(target_X)
        test_dcv = self.compute_dcv(test_dist, target_X.shape[0])
        # for each project
        pred_list = []
        for proj_name in set(self.train_pool_domain):
            source_X =self.train_pool_X[np.array(self.train_pool_domain) \
                == proj_name]
            source_y = self.train_pool_y[source_X.index]
            if source_X.shape[0] == 1:
                continue
            else:
                pred_list.append(self.predict_one_proj(source_X, source_y, 
                    target_X, test_dist, test_dcv))

        # use vote counting to determine confidence and prediction
        votes = zip(*pred_list)
        confidence_arr = np.mean(votes, 1)
        predictions_arr = confidence_arr > 0.5
        return (confidence_arr, np.array(predictions_arr))

    def json_encode(self):
        base = tl_alg.Base_Transfer.json_encode(self)
        base.update({"dims" : self.dims, 
            "kernel_type" : self.kernel_type, 
            "kernel_param" : self.kernel_param,
            "mu" : self.mu})

        return base

