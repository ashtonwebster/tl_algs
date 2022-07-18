import numpy as np
import pandas as pd
import json
from . import tl_alg


def sim_minmax(column):
    """Similarity score using the range between min and max
    for a value

    Args:
      column: a given feature column

    Returns:
      tuple: A tuple of the form (min, max)

    """
    return min(column), max(column)


def sim_std(column):
    """Similarity score using the standard error for a column

    Args:
      column: a given feature column

    Returns:
      tuple: tuple with the first element one std dev below the mean
      and the second element one std dev above the mean

    """
    return (np.mean(column) - np.std(column), np.mean(column) + np.std(column))


class TransferNaiveBayes(tl_alg.Base_Transfer):
    """
    Transfer Naive Bayes algorithm, as described by Ma [1]

    Args:
        similarity_func: the function used to determine if two features
                          are similar
        num_disc_bins: the number of bins to use for discretization
        discretize: whether to discretize (if passing in already categorical
                    data, set this to false.

    Returns:
        A classifier

    [1] Ma, Y., Luo, G., Zeng, X., & Chen, A. (2012). Transfer learning
    for cross-company software defect prediction. Information and 
    Software Technology, 54(3), 248-256.
    https://doi.org/10.1016/j.infsof.2011.09.007
    """

    def __init__(
            self,
            test_set_X,
            test_set_domain,
            train_pool_X,
            train_pool_y,
            train_pool_domain,
            rand_seed=None,
            classifier_params={},
            similarity_func=sim_std,
            num_disc_bins=10,
            alpha=1,
            discretize=True):

        super(
            TransferNaiveBayes,
            self).__init__(
            test_set_X,
            test_set_domain,
            train_pool_X,
            train_pool_y,
            train_pool_domain,
            None,
            rand_seed=rand_seed,
            classifier_params=classifier_params)

        self.similarity_func = similarity_func
        self.num_disc_bins = num_disc_bins
        self.discretize = discretize
        self.alpha = alpha

        self.cached_n_j = {}
        self.cached_n_c = None
        self.cached_cond_prob = {}

    def isinrange(self, row, ranges):
        """ returns a boolean vector where the ith entry corresponds
        to whether the ith feature is within the range specified in ranges

        Args:
            row: The row to check
            ranges: a list of the form [(low, high),...] for each feature

        Returns:
            A boolean vector

        """
        return sum([ranges[i][0] <= cell <= ranges[i][1] for i, cell in row.items()])

    def get_weights(self):
        """
        Gets the prior distirbutions for the features (aka the weights)
        """
        # ranges is analogous to min_ij and max_ij in the paper
        ranges = self.test_set_X.apply(self.similarity_func)
        # for each row, compute similarity, analogous to s_i in the paper

        similarity_count = self.train_pool_X.apply(lambda x: self.isinrange(x, ranges), axis=1)
        # for each row, calculate weight, analogous to w_i in the paper
        # note that in the paper the number of features (train_pool_X.shape[0]) is k
        weight_vec = map(lambda x: float(
            float(x) / (self.train_pool_X.shape[1] - x + 1)**2), similarity_count)
        return list(weight_vec)

    def get_discretized_X(self):
        """
        Creates discretized versions of the training and test instances.  Test instances are
        used as the "baseline" which dictates the bins.  The training instances are then 
        discretized using the same bin edges.  The number of bins is controlled by the
        self.num_disc_bins parameter.

        Returns:
            (X_train_disc, X_test_disc)
        """
        test_disc_arr, train_disc_arr = [], []
        for col_ind in range(self.test_set_X.shape[1]):
            # start with discretizing test set, save bins
            try:
                labels = list(map(str, range(self.num_disc_bins)))
                test_disc, bins = pd.cut(self.test_set_X.iloc[:, col_ind], 
                            bins = self.num_disc_bins, 
                            labels = labels,
                            retbins=True)
            except ValueError:
                test_disc, bins = pd.cut(self.test_set_X.iloc[:, col_ind],
                            bins = [float('inf') * -1, float('inf')],
                            labels = [str(0)],
                            retbins=True)

            # makde sure bins cover entire interval
            bins[0] = -1 * float('inf')
            bins[-1] = float('inf')
            # use (modified) test set bins for training set discretization
            labels = list(map(str, range(len(bins)-1)))
            train_disc = pd.cut(self.train_pool_X.iloc[:, col_ind], 
                            bins=bins, 
                            labels=labels)
            test_disc_arr.append(test_disc)
            train_disc_arr.append(train_disc)
        # combine discretized series to data frame
        return pd.concat(train_disc_arr, axis=1), pd.concat(test_disc_arr, axis=1)


    def get_cached_n_j(self, feature_index, X_weighted):
        if feature_index not in self.cached_n_j.keys():
            self.cached_n_j[feature_index] = len(X_weighted.iloc[:, feature_index].unique())
        return self.cached_n_j[feature_index]

    def get_cached_n_c(self):
        if not self.cached_n_c:
            self.cached_n_c = len(self.train_pool_y.unique())
        return self.cached_n_c

    def get_cached_conditional_prob(self, label, feature_index, feature_val,
            X_weighted, n_c, n_j, alpha):
        if (label, feature_index, feature_val) not in self.cached_cond_prob.keys():
            feature_mask = np.asarray(self.train_pool_y == label).reshape(-1) & \
                np.asarray(X_weighted.iloc[:, feature_index] == feature_val)\
                .reshape(-1)
            class_mask = np.asarray(self.train_pool_y == label).reshape(-1)
            self.cached_cond_prob[(label, feature_index, feature_val)] = \
                    (X_weighted.myweight.loc[feature_mask].sum() + alpha) / \
                    (X_weighted.myweight.loc[class_mask].sum() + n_c * alpha)
        return self.cached_cond_prob[(label, feature_index, feature_val)]

    def get_class_prob(self, X_weighted, label, alpha):
        """ Computes P(C) according to equation 7 in [1] 
        
        Args:
            X_weighted: dataframe with columns of the form [feature1, feature2,
                        ..., weight]
            label: the label to calculate the probability of
            alpha: the laplace smoothing factor, default is 1
        """
        # number of classes
        n_c = len(self.train_pool_y.unique())
        mask = np.asarray(self.train_pool_y == label).reshape(-1)
        return (X_weighted[mask].myweight.sum() + alpha) / \
                    (X_weighted.myweight.sum() + n_c * alpha)

    def get_conditional_prob(self, X_weighted, label, feature_index, 
            feature_val, alpha):
        """ Computes P(a_j|c), where a_j is value j for feature a, and 
        c is the class

        Calculated according to equation 8 from [1]

        Args:
            X_weighted: dataframe with columns of the form [feature1, feature2,
                        ..., weight]
            label: the label to calculate the probability of
            feature_val: the value of the feature to calculate the prob. of
            alpha: the laplace smoothing factor, default is 1
        """
        n_j = self.get_cached_n_j(feature_index, X_weighted)
        n_c = self.get_cached_n_c()
        return self.get_cached_conditional_prob(label, feature_index, feature_val,
                X_weighted, n_c, n_j, alpha)

    
    def get_posterior_prob(self, X_weighted, label, instance, alpha):
        """ Compute P(c|u), where c is the class and u is the train instance

        Calculated according to equation 1 from [1]

        Args:
            label: the label to calculate the probability of
            instance: a row from the training set

        Returns:
            P(c|u)
        """

        # building numerator
        numerator = self.get_class_prob(X_weighted, label, alpha)
        # column index = j
        for j in range(len(instance)):
            numerator *= self.get_conditional_prob(X_weighted,
                    label = label, feature_index = j, feature_val=instance[j],
                    alpha=alpha)

        # building denominator
        denominator = 0
        for c in self.train_pool_y.unique():
            term = self.get_class_prob(X_weighted, c, alpha)
            for j in range(len(instance)):
                term *= self.get_conditional_prob(X_weighted,
                        label = c, feature_index = j, 
                        feature_val=instance[j],
                        alpha=alpha)
            denominator += term

        
        return numerator / denominator if denominator != 0 else 0


    def train_filter_test(self):
        """Applies weight filter and returns predictions"""
        weights = self.get_weights()

        if self.discretize:
            X_train_disc, X_test_disc = self.get_discretized_X()
        else:
            X_train_disc, X_test_disc = self.train_pool_X, self.test_set_X

        X_train_disc['myweight'] = weights
        
        y_pred, y_conf = [], []
        for __, row in X_test_disc.iterrows():
            class_probs = [self.get_posterior_prob(X_train_disc,
                c, row, self.alpha) for c in [False, True]]
            i = np.argmax(class_probs)
            y_pred.append(i == 1)
            # always get probability of positive prediction
            y_conf.append(class_probs[1])
        return np.array(y_conf), np.array(y_pred)
            

    def json_encode(self):
        """Encodes this class as a json object"""
        base = tl_alg.Base_Transfer.json_encode(self)
        base.update({"similarity_func": self.similarity_func.__name__,
            "num_disc_bins" : self.num_disc_bins,
            "discretize" : self.discretize
            })
        return base
