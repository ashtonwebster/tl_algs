import numpy as np
import pandas as pd
import json
from tl_algs import tl_alg
from vuln_toolkit.common import vuln_metrics
from sklearn.dummy import DummyClassifier


# ----------------Metrics----------------

def acc_metric(clf, X_test, y_test):
    '''Metric for accuracy'''
    return clf.score(X_test, y_test.tolist())


def aucec50_metric(clf, X_test, y_test):
    '''Metric for aucec50'''
    confidence_matrix = clf.predict_proba(X_test)
    # sometimes there are only confidence estimates for the negative class
    # because all predictions are negative
    confidence = [a[1] for a in confidence_matrix] \
        if len(confidence_matrix[0]) == 2 \
        else [0 for _ in confidence_matrix]
    return vuln_metrics.aucec_50_score(y_test.tolist(), confidence)


def aucec50_raw_metric(y_actual, y_predicted, y_confidence):
    return vuln_metrics.aucec_50_score(y_actual.tolist(), y_confidence)


# ----------------Voters----------------

def count_vote(F_star, test_set_X, threshold_prop=0.5):
    '''Takes the list of selected weak classifiers and return 
    (confidence,prediction)
    threshold: proportion of functions which must predict yes for the label output to be yes
    '''
    pred_list = []
    for f in F_star:
        y_pred = f.predict(test_set_X)
        pred_list.append(y_pred.tolist())
    votes = zip(*pred_list)
    threshold_count = len(F_star) * threshold_prop

    # confidence = num votes / total
    confidence_arr = [float(sum(instance_votes)) / len(F_star)
                      for instance_votes in votes]
    # prediction = confidence > threshold_count
    prediction_arr = [confidence >
                      threshold_count for confidence in confidence_arr]
    return (confidence_arr, prediction_arr)


def mean_confidence_vote(F_star, test_set_X):
    pred_list = []
    for f in F_star:
        proba = f.predict_proba(test_set_X)
        # give the confidence for the positive class (vulnerable), unless there
        # is no confidence
        y_pred = [a[-1] for a in proba]
        pred_list.append(y_pred)
        #print("y_pred", str(y_pred))

    # combine so that each row is a list of predictions s.t. the ith row and the jth element in that row is the
    # jth prediction of the ith test instance
    votes = zip(*pred_list)
    # print(votes[0])
    mean_confidence = map(np.mean, votes)
    # return the mean confidence and if the confidence is greater than .5,
    # predict vulnerability
    return (mean_confidence, map((lambda x: x >= .5), mean_confidence))


# ----------------Filters----------------

def no_filter(f_0, F, X_target, y_target):
    '''Apply no filter to the data, just merge F and F_0 and return'''
    F.append(f_0)
    return F


def all_filter(f_0, F, X_target, y_target):
    '''Return only the fallback classifier'''
    return [f_0]


def sc_trbag_filter(f_0, F, X_target, y_target, metric=aucec50_metric):
    ''' filter based on SC method and return F_star (filtered weak classifiers)
    f_0: baseline
    let metric be a function of the form f(clf, X_test, y_test) -> real number
    '''
    fallback_performance = metric(f_0, X_target, y_target)
    F_star = [f_0]
    # print(fallback_performance)
    for f in F:
        # append f if it has a better performance on the target set than the
        # fallback f_0
        m = metric(f, X_target, y_target)
        # print(m)
        if m > fallback_performance:
            F_star.append(f)
    #print("f_star len", len(F_star))
    return F_star


def mvv_filter(
        f_0,
        F,
        X_target,
        y_target,
        raw_metric=aucec50_raw_metric,
        vote_func=mean_confidence_vote):
    '''
    '''
    fallback_predicted = f_0.predict(X_target)
    proba = f_0.predict_proba(X_target)
    # give the confidence for the positive class (vulnerable), unless there is
    # no confidence
    fallback_confidence = [a[1] for a in proba] if len(proba[0]) > 1 else [
        0 for a in proba]
    fallback_performance = raw_metric(
        y_target,
        list(fallback_predicted),
        list(fallback_confidence))
    #print("fallback performance: " + str(fallback_performance))
    F_star = [f_0]
    # sort by descending metric scores (assume higher metric scores are better)
    proba = f.predict_proba(X_target)
    F_scores = [
        raw_metric(
            y_target, f.predict(X_target), [
                a[1] for a in proba] if len(
                proba[0]) > 1 else [
                    0 for a in proba]) for f in F]
    # print(F_scores)
    # zip the lists so we have [(classifier, score), ...]
    F_merged = zip(F, F_scores)

    # sort clasifiers by score
    F_merged_sorted = sorted(F_merged, lambda a, b: int(a[1] - b[1]))
    F_sorted = [a[0] for a in F_merged_sorted]
    for f in F_sorted:
        # deep copy F_star so changes to F_candidate do not affect f_star
        F_candidate = list(F_star)
        #print("before append F_candidate len: {0}, F_star len: {1}".format(len(F_candidate), len(F_star)))
        # append a new classifier to the set of possible learners
        F_candidate.append(f)
        #print("after append F_candidate len: {0}, F_star len: {1}".format(len(F_candidate), len(F_star)))

        # vote among F_candidate learners
        confidence, predictions = vote_func(F_candidate, X_target)
        #print("F_candidate confidence for each: " + str(confidence[:5]))
        # get performance of new candidate set
        candidate_performance = raw_metric(y_target, predictions, confidence)
        #print("candidate performance: " + str(candidate_performance))
        # only select candidate set if its performance is better than the
        # fallback classifier alone
        if candidate_performance > fallback_performance:
            #print("candidate performance: {0} fallback performance {1}".format(candidate_performance, fallback_performance))
            F_star = list(F_candidate)
            fallback_performance = candidate_performance

    #print("f star length: " + str(len(F_star)))
    return F_star


class TrBag(tl_alg.Base_Transfer):

    def __init__(self, test_set_X, test_set_domain, train_pool_X,
                 train_pool_y, train_pool_domain,
                 Base_Classifier, rand_seed=None, classifier_params={},
                 T=100, bag_prop=1.0, filter_func=sc_trbag_filter,
                 vote_func=count_vote, validate_proportion=None):
        """ Apply the trBag method of transfer learning and return a tuple of the form (confidence_array, predicted_classes)
        T: number of classifiers to bootstrap on (plus one baseline classifier)

        bag_prop: bag size should be this proportion of trainin set size, on the range [0,1]

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
        """

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
        self.bag_prop = bag_prop
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
        """ split the labeled data into validation and training and return
        (X_target_train, y_target_tain, X_validate, y_validate, train_pool_X, train_pool_y)
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
            #print("Validation size", str(X_validate.shape[0]))
            assert X_validate.shape[0] == len(y_validate)
            # Remove validation set from target training set
            # we are able to use iloc (position based indexing) because we
            # reset the index above
            X_target_train = X_target.ix[
                X_target.index.difference(
                    X_validate.index), ]
            y_target_train = y_target[X_target_train.index]
            #print("Train size", str(X_target_train.shape[0]))
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
            T,
            bag_prop,
            rand_seed,
            classifier_params):
        """ Bootstrap (sample with replacement) to create T classifiers and return list of classifiers F

        T: number of classifiers
        bag_prop: the proportion of available training data to use
        """

        F = []
        for i in range(0, T):
            f = Base_Classifier(random_state=rand_seed, **classifier_params)
            # sample with replacement
            X_bootstrap = train_pool_X.sample(
                bag_prop * len(train_pool_X),
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
            classifier_params={},
            T=100,
            bag_prop=1.0,
            filter_func=sc_trbag_filter,
            vote_func=count_vote,
            validate_proportion=None,
            rand_seed=None):
        ''' Apply the trBag method of transfer learning and return a tuple of the form (confidence_array, predicted_classes)
        T: number of classifiers to bootstrap on (plus one baseline classifier)

        bag_prop: bag size should be this proportion of trainin set size, on the range [0,1]

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
        '''

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
            T,
            bag_prop,
            rand_seed,
            classifier_params)

        # create baseline learning which uses only labeled target data
        if (X_target_train.shape[0] == 0):
            # if no training instances, use dummy classifier
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
        return self.trbag_validate(
            self.test_set_X,
            self.test_set_domain,
            self.train_pool_X,
            self.train_pool_y,
            self.train_pool_domain,
            self.Base_Classifier,
            classifier_params=self.classifier_params,
            T=self.T,
            bag_prop=self.bag_prop,
            filter_func=self.filter_func,
            vote_func=self.vote_func,
            validate_proportion=self.validate_proportion,
            rand_seed=self.rand_seed)

    def json_encode(self):
        base = tl_alg.Base_Transfer.json_encode(self)
        base.update({"T": self.T,
                     "bag-prop": self.bag_prop,
                     "filter_func": self.filter_func.__name__,
                     "vote_func": self.vote_func.__name__,
                     "validate_prop": self.validate_proportion})
        return base
