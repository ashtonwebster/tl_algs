import numpy as np
import pandas as pd
import json
from tl_algs import tl_alg
from tl_algs import voter
# from vuln_toolkit.common import vuln_metrics
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score 




# ----------------Filters----------------

def no_filter(f_0, F, X_target, y_target):
    """Apply no filter to the data, just merge F and F_0 and return

    Args:
      f_0: baseline learner
      F: set of possible learners
      X_target: target training instances
      y_target: target training labels

    Returns:
      list: F with f_0 added to it

    """
    F.append(f_0)
    return F


def all_filter(f_0, F, X_target, y_target):
    """Filter all other learners except the basline

    Args:
      f_0: baseline learner
      F: set of possible learners
      X_target: target training instances
      y_target: target training labels

    Returns:
      list: An array just containing f_0

    """
    return [f_0]


def sc_trbag_filter(f_0, F, X_target, y_target, metric=f1_score):
    """filter based on SC method and return F_star (filtered weak classifiers)
    f_0: baseline

    Args:
      f_0: baseline learner
      F: set of possible learners
      X_target: target training instances
      y_target: target training labels
      metric: a function of the form f(y_true, y_pred) -> real number
    (Default value = sklearn.metrics.f1_score)

    Returns:
      list: Filtered set of leaners F_star

    """
    fallback_performance = metric(y_target, f_0.predict(X_target))
    F_star = [f_0]
    # print(fallback_performance)
    for f in F:
        # append f if it has a better performance on the target set than the
        # fallback f_0
        y_pred = f.predict(X_target)
        m = metric(y_target, y_pred)

        # print(m)
        if m > fallback_performance:
            F_star.append(f)
    # print("f_star len", len(F_star))
    return F_star


def mvv_filter(
        f_0,
        F,
        X_target,
        y_target,
        raw_metric=f1_score,
        vote_func=voter.mean_confidence_vote):
    """Majority Voting on the Validation Set (see
    https://doi.org/10.1109/ICDM.2009.9 for details)
    
    If X_target is the entire target set, this is actually Maximum Voting on
    the Target set (MVT).  If X_target is a subset of the target data (i.e.
    a validation set), then this is Majority Voting on the Validation Set (MVV)

    Args:
      f_0: baseline learner
      F: set of possible learners
      X_target: target training instances
      y_target: target training labels
      raw metric: a function of the form f(y_true, y_pred) -> real number
    (Default value = sklearn.metrics.f1_score)
      vote_func: a function of the form f(F_star, test_set_X) -> (confidences,
    predictions) (Default value = mean_confidence_vote)

    Returns:
      list: A filtered list of learners F_star

    """
    fallback_predicted = f_0.predict(X_target)
    fallback_performance = raw_metric(y_target, fallback_predicted)
    F_star = [f_0]
    # sort by descending metric scores (assume higher metric scores are better)
    F_scores = [raw_metric(y_target, f.predict(X_target)) for f in F]
    # zip so we have [(classifier, score), ...]
    F_merged = zip(F, F_scores)
    F_merged_sorted = sorted(F_merged, key=lambda x: x[1], reverse=True)
    # traverse list of F by descending score
    for f, __ in F_merged_sorted:
        # shallow copy F_star so changes to F_candidate do not affect f_star
        F_candidate = list(F_star)
        # append a new classifier to the set of possible learners
        F_candidate.append(f)
        # vote among F_candidate learners
        confidence, predictions = vote_func(F_candidate, X_target)
        # get performance of new candidate set
        candidate_performance = raw_metric(y_target, predictions)
        # only select candidate set if its performance is better than the
        # fallback classifier alone
        if candidate_performance > fallback_performance:
            F_star = list(F_candidate)
            fallback_performance = candidate_performance

    return F_star


class TrBag(tl_alg.Base_Transfer):
    """Apply the trBag method of transfer learning and return a tuple of the form (confidence_array, predicted_classes)
        T: number of classifiers to bootstrap on (plus one baseline classifier)
    
        sample_size: The number of instances to bootstrap sample (should be equal to the number of 
                test target instances
    
        filter_func: which filter function to use (i.e. )
    
        classifier_params: dictionary of parameters to be passed to the base classifier.  Do not include rand seed; instead use the
        rand seed parameter and it will be passed to the classifier
    
        validate_proportion: Default is None, which means validation set is not used.
        Otherwise, this indicates how much of the target training data to use for validation.  For example, if the target
        domain has 100 instances total, 50 of which are used for testing, then there are a total of 50 remaining for "training".
        Among these 50 training, a validation proportion of .5 would use 25 for validation and 25 for development.
    
        random_seed: A seed for the random number generator (used for replication of experiments).  This will be passed as a
        base classifier param, do not pass it again
    
    
        see: https://doi.org/10.1109/ICDM.2009.9

    Args:

    Returns:

    """

    def __init__(self, test_set_X, test_set_domain, train_pool_X,
                 train_pool_y, train_pool_domain,
                 Base_Classifier, sample_size, rand_seed=None, classifier_params={},
                 T=25, filter_func=sc_trbag_filter,
                 vote_func=voter.count_vote, validate_proportion=None):
        
        # this should be passing all the base parameters to the superclass
        super(
            TrBag,
            self).__init__(
            test_set_X,
            test_set_domain,
            train_pool_X,
            train_pool_y,
            train_pool_domain,
            Base_Classifier,
            rand_seed=rand_seed,
            classifier_params=classifier_params)
        # define the rest of the parameters here.
        self.T = T
        self.sample_size = sample_size 
        self.filter_func = filter_func
        self.vote_func = vote_func

        self.validate_proportion = validate_proportion
# ----------------Driver----------------

    def split_validation(
            self,
            test_set_domain,
            train_pool_X,
            train_pool_y,
            train_pool_domain,
            validate_proportion,
            rand_seed):
        """split the labeled data into validation and training and return
        (X_target_train, y_target_tain, X_validate, y_validate, train_pool_X, train_pool_y)

        Args:
          test_set_domain: domain of the test set
          train_pool_X: training instances
          train_pool_y: training instance labels
          train_pool_domain: training instance domains as a list, such that the i'th element
        of the list corresponds to the i'th training instance
          validate_proportion: number from 0 to 1 proportion of training instances
        to use for validation
          rand_seed: Seed for consistent random results, or None

        Returns:
          tuple: A tuple of the form X_target_train, y_target_train, X_validate,
          tuple: A tuple of the form X_target_train, y_target_train, X_validate,
          y_validate, train_pool_X, train_pool_y)

        """

        if (validate_proportion is None):
            # If not using validation set, then train set = validation set
            # Get all target instances
            X_target = train_pool_X[
                np.array(train_pool_domain) == test_set_domain]
            y_target = train_pool_y[
                np.array(train_pool_domain) == test_set_domain]
            X_validate = X_target_train = X_target
            y_validate = y_target_train = y_target
        else:
            # If using validation, then train set is all the data not in the validation set
            # Get all target instances
            X_target = train_pool_X[
                np.array(train_pool_domain) == test_set_domain]
            y_target = train_pool_y[
                np.array(train_pool_domain) == test_set_domain]
            # Create validation set
            X_validate = X_target.sample(
                frac=validate_proportion, random_state=rand_seed)
            y_validate = y_target[X_validate.index]
            # print("Validation size", str(X_validate.shape[0]))
            assert X_validate.shape[0] == len(y_validate)
            # Remove validation set from target training set
            # we are able to use iloc (position based indexing) because we
            # reset the index above
            X_target_train = X_target.ix[
                X_target.index.difference(
                    X_validate.index), ]
            y_target_train = y_target[X_target_train.index]
            # print("Train size", str(X_target_train.shape[0]))
            assert X_target_train.shape[0] == len(y_target_train)
            # Remove validation from train pool
            train_pool_X = train_pool_X.ix[
                train_pool_X.index.difference(
                    X_validate.index), ]
            train_pool_y = train_pool_y[train_pool_X.index]
        return (
            X_target_train,
            y_target_train,
            X_validate,
            y_validate,
            train_pool_X,
            train_pool_y)

    def bootstrap(
            self,
            train_pool_X,
            train_pool_y,
            Base_Classifier,
            sample_size,
            T,
            rand_seed,
            classifier_params):
        """Bootstrap (sample with replacement) to create T classifiers and return list of classifiers F
        
          train_pool_X: Training instances
          train_pool_y: training instance labels
          Base_Classifier: the classifier to be used
          sample_size: The number of instances to bootstrap sample (should be equal to the number of 
                test target instances
          T: number of classifiers
          rand_seed: random seed or None
          classifier_params: Classifier parameters passed as dictionary

        Returns:
          list: List of potential learners F

        """

        F = []
        for i in range(0, T):
            f = Base_Classifier(random_state=rand_seed, **classifier_params)
            # sample with replacement
            X_bootstrap = train_pool_X.sample(
                n=sample_size,
                replace=True,
                random_state=rand_seed + i)
            # print(len(X_bootstrap.index.unique()))
            y_bootstrap = train_pool_y[X_bootstrap.index]
            f.fit(X_bootstrap, y_bootstrap.tolist())
            F.append(f)

        return F

    def trbag_validate(
            self,
            test_set_X,
            test_set_domain,
            train_pool_X,
            train_pool_y,
            train_pool_domain,
            Base_Classifier,
            sample_size,
            classifier_params={},
            T=25,
            filter_func=sc_trbag_filter,
            vote_func=voter.count_vote,
            validate_proportion=None,
            rand_seed=None):
        """Apply the trBag method of transfer learning and return a tuple of the form (confidence_array, predicted_classes)
        
        see: https://doi.org/10.1109/ICDM.2009.9

        Args:
          test_set_X: 
          test_set_domain: 
          train_pool_X: 
          train_pool_y: 
          train_pool_domain: 
          Base_Classifier: Classifier function to use (i.e. Random Forest)
          sample_size: the number of instances to sample in the bagging phase.  This should be equal to the minimum of the target test set
          and training set.
          classifier_params: dictionary of parameters to be passed to the base classifier.  Do not include rand seed; instead use the
        rand seed parameter and it will be passed to the classifier (Default value = {})
          T: number of classifiers to bootstrap on (plus one baseline classifier) (Default value = 100)
          filter_func: which filter function to use (i.e. sc_trbag_filter)  (Default value = sc_trbag_filter)
          vote_func: (Default value = count_vote)
          validate_proportion: Default is None, which means validation set is not used.
        Otherwise, this indicates how much of the target training data to use for validation.  For example, if the target
        domain has 100 instances total, 50 of which are used for testing, then there are a total of 50 remaining for "training".
        Among these 50 training, a validation proportion of .5 would use 25 for validation and 25 for development. (Default value = None)
          random_seed: A seed for the random number generator (used for replication of experiments).  This will be passed as a
        base classifier param, do not pass it again
          rand_seed: (Default value = None)

        Returns:
          tuple: tuple of the form (confidences, predictions) for each instance

        """

        (X_target_train,
         y_target_train,
         X_validate,
         y_validate,
         train_pool_X,
         train_pool_y) = self.split_validation(test_set_domain,
                                               train_pool_X,
                                               train_pool_y,
                                               train_pool_domain,
                                               validate_proportion,
                                               rand_seed)

        # bootstrap sample to generate T learners
        F = self.bootstrap(
            train_pool_X,
            train_pool_y,
            Base_Classifier,
            sample_size,
            T,
            rand_seed,
            classifier_params)

        # create baseline learning which uses only labeled target data
        if (X_target_train.shape[0] == 0 or validate_proportion is None
                or  validate_proportion == 0):
            # if no target training instances or nothing to
            # validate on, use dummy classifier
            # in this case, no target-domain data is used for training at all.
            f_0 = DummyClassifier(
                strategy='uniform',
                random_state=rand_seed).fit(
                train_pool_X,
                train_pool_y.tolist())
        else:
            f_0 = Base_Classifier(random_state=rand_seed,
                                  **classifier_params).fit(X_target_train,
                                                           y_target_train.tolist())

        # filter
        F_star = filter_func(f_0, F, X_validate, y_validate)

        # return count_vote(F_star, test_set_X)
        return vote_func(F_star, test_set_X)

    def train_filter_test(self):
        """Apply the trBag method of transfer learning and return a tuple of the form (confidence_array, predicted_classes)"""
        return self.trbag_validate(
            self.test_set_X,
            self.test_set_domain,
            self.train_pool_X,
            self.train_pool_y,
            self.train_pool_domain,
            self.Base_Classifier,
            self.sample_size,
            classifier_params=self.classifier_params,
            T=self.T,
            filter_func=self.filter_func,
            vote_func=self.vote_func,
            validate_proportion=self.validate_proportion,
            rand_seed=self.rand_seed)

    def json_encode(self):
        """Encode this classifier as a json object"""
        base = tl_alg.Base_Transfer.json_encode(self)
        base.update({"T": self.T,
                     "sample_size": self.sample_size,
                     "filter_func": self.filter_func.__name__,
                     "vote_func": self.vote_func.__name__,
                     "validate_prop": self.validate_proportion})
        return base
