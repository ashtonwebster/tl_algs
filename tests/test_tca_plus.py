import pandas as pd
import numpy as np
from tl_algs import tca_plus

def test_minimax_normalization_ones():
    _tca = tca_plus.TCAPlus(test_set_X=pd.DataFrame(np.ones((5,5))),
            test_set_domain=1,
            train_pool_X = pd.DataFrame(np.ones((5,5))),
            train_pool_y=None,
            train_pool_domain=[0,0,0,1,1],
            Base_Classifier=None)
    print(_tca.minmax_normalization())

def test_normalization():
    source_X= pd.DataFrame(np.random.rand(5,5))
    target_X = pd.DataFrame(np.random.rand(5,5))

    _tca = tca_plus.TCAPlus(test_set_X=None,
            test_set_domain=1,
            train_pool_X = None,
            train_pool_y=None,
            train_pool_domain=[0,0,0,1,1],
            Base_Classifier=None)
    #print(_tca.minmax_normalization(source_X, target_X))
    print(_tca.zscore_normalization(source_X, target_X))
    print(_tca.zscore_source_normalization(source_X, target_X))
    print(_tca.zscore_target_normalization(source_X, target_X))

def test_distance():
    _tca = tca_plus.TCAPlus(test_set_X=None,
        test_set_domain=None,
        train_pool_X = None, 
        train_pool_y=None,
        train_pool_domain=None,
        Base_Classifier=None)
    dist_ = _tca.compute_distance_set(np.random.rand(4,50))
    print(_tca.compute_dcv(dist_, 4))

def test_similarity():
    _tca = tca_plus.TCAPlus(test_set_X=None,
        test_set_domain=None,
        train_pool_X = None, 
        train_pool_y=None,
        train_pool_domain=None,
        Base_Classifier=None)
    assert _tca.compute_comp_similarity(905, 238) == tca_plus.SimLevel.MUCH_LESS
    assert _tca.compute_comp_similarity(60, 60) == tca_plus.SimLevel.SAME
    assert _tca.compute_comp_similarity(59, 60) == tca_plus.SimLevel.SAME
    assert _tca.compute_comp_similarity(100, 125) == tca_plus.SimLevel.SLIGHTLY_MORE
    assert _tca.compute_comp_similarity(100, 39) == tca_plus.SimLevel.MUCH_LESS

def test_dcv_sim_vector():
    _tca = tca_plus.TCAPlus(test_set_X=None,
        test_set_domain=None,
        train_pool_X = None, 
        train_pool_y=None,
        train_pool_domain=None,
        Base_Classifier=None)
    dist1 = _tca.compute_distance_set(np.random.rand(4,50))
    dist2 = _tca.compute_distance_set(np.random.rand(4,50))
    dcv_1, dcv_2 = _tca.compute_dcv(dist1, 4), _tca.compute_dcv(dist2, 4)
    print(_tca.compute_dist_similarity(dcv_1, dcv_2))

def test_full():
    X_test = pd.DataFrame(np.random.rand(5,5))
    X_train = test_set_X=pd.DataFrame(np.ones((5,5)))
    _tca = tca_plus.TCAPlus(X_test,
            test_set_domain=1,
            train_pool_X = X_train,
            train_pool_y=None,
            train_pool_domain=[0,0,0,1,1],
            Base_Classifier=None)
    dist1 = _tca.compute_distance_set(X_test)
    dist2 = _tca.compute_distance_set(X_train)
    dcv_1, dcv_2 = _tca.compute_dcv(dist1, 4), _tca.compute_dcv(dist2, 4)

    print(_tca.minmax_normalization())








#test_minimax_normalization_ones()
#test_normalization()
#test_distance()
#test_similarity()
#test_dcv_sim_vector()
