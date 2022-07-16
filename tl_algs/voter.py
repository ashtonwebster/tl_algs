import numpy as np
import pandas as pd

def count_vote(clf_list, test_set_X, threshold_prop=0.5):
    """Takes the list of selected weak classifiers and return
    (confidence,prediction)
    threshold: proportion of functions which must predict yes for the label
    output to be yes

    Args:
      clf_list: set of classifiers to vote
      test_set_X: Test set training instances
      threshold_prop: Proportion of votes required to assign positive label
    (Default value = 0.5)

    Returns:
      tuple: Tuple of the form (confidences, predictions)

    """
    pred_list = []
    for f in clf_list:
        y_pred = f.predict(test_set_X)
        pred_list.append(y_pred.tolist())

    # votes = zip(*pred_list)
    votes = np.array(pred_list)
    # confidence = num votes / total
    confidence_arr = np.mean(votes, 1)
    # prediction = confidence > threshold_count

    prediction_arr = confidence_arr > threshold_prop
    return (confidence_arr, np.array(prediction_arr))



def mean_confidence_vote(clf_list, test_set_X, threshold_prop=0.5):
    """
    Sums the confidence of each predictor and uses that as the vote.
    Args:
      clf_list: set of classifiers to vote
      test_set_X: Test set training instances

    Returns:
      tuple: Tuple of the form (confidences, predictions)

    """

    pred_list = []
    for f in clf_list:
        proba = f.predict_proba(test_set_X)
        # give the confidence for the positive class (vulnerable), unless there
        # is no confidence
        y_pred = [a[1] if len(a) > 1 else 0 for a in proba]
        pred_list.append(y_pred)
        # print("y_pred", str(y_pred))

    # combine so that each row is a list of predictions s.t. the ith row and the jth element in that row is the
    # jth prediction of the ith test instance
    confidence_votes = zip(*pred_list)
    # print(votes[0])
    mean_confidence = np.mean(confidence_votes, 1)
    predictions = mean_confidence > threshold_prop

    # return the mean confidence and if the confidence is greater than .5,
    # predict vulnerability
    return (mean_confidence, predictions)

