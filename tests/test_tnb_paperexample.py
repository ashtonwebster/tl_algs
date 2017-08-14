# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
from tl_algs import tnb
RAND_SEED = 2016 
random.seed(RAND_SEED) # change this to see new random data!

"""
This example is taken from the paper [1].  Not all results are the same
due to arithmetic errors in the paper.

    [1] Ma, Y., Luo, G., Zeng, X., & Chen, A. (2012). Transfer learning
    for cross-company software defect prediction. Information and 
    Software Technology, 54(3), 248-256.
    https://doi.org/10.1016/j.infsof.2011.09.007

"""

X_train = pd.DataFrame([[2,1,3], [1,2,2], [1,3,4]])
X_test = pd.DataFrame([[2,1,3], [1,2,3]])
y_train = pd.Series([False, False, True])

# data is already discretized so we specify discretize=False
w = tnb.TransferNaiveBayes(test_set_X=X_test, 
        test_set_domain='a', 
        train_pool_X=X_train, 
        train_pool_y=y_train, 
        train_pool_domain=None,
        rand_seed=None, 
        similarity_func=tnb.sim_minmax,
        discretize=False)

conf, y_pred = w.train_filter_test()
# 0.936 obtained by manual computation
assert abs(conf[0] - 0.936) < 0.001
