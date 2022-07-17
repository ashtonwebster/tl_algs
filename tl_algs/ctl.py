from tl_algs import tl_alg
from sklearn.cluster import KMeans

class ClusterThenLabel(tl_alg.Base_Transfer):
    """
    This transfer learning algorithm clusters the data using k means
    and then labels using the provided classifier
    """
    
    def __init__(self, test_set_X, test_set_domain, train_pool_X, train_pool_y,
                 train_pool_proj, Base_Classifier, classifier_params={},
                 rand_seed=None, num_clusters=8):
        
        super(ClusterThenLabel, self).__init__(
            test_set_X,
            test_set_domain,
            train_pool_X, 
            train_pool_y,
            train_pool_proj, 
            Base_Classifier,
            rand_seed=rand_seed,
            classifier_params=classifier_params
        )

        self.num_clusters = num_clusters
  
     
    def train_filter_test(self):
        """
        Train classifier on filtered training data using the k-means heuristic
        and return class predictions and confidence values for each prediction.

        Returns:
            confidence: List of confidence values, the ith entry of which gives
                the confidence for the ith prediction.
            predictions: List of class predictions.
        """
        X_train_pool, y_train_pool, X_test = (self.train_pool_X.copy().reset_index(drop=True),
                self.train_pool_y.copy().reset_index(drop=True),
                self.test_set_X.copy().reset_index(drop=True)
                )

        # build kmeans clusters on training data
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.rand_seed)\
                    .fit(X_train_pool)
        cluster_labels = kmeans.predict(X_train_pool)
        
        # build models for each cluster
        models = []
        for i in range(self.num_clusters):
            mask = i == cluster_labels
            models.append(self.Base_Classifier().fit(X_train_pool[mask],
                list(y_train_pool[mask])))

        # get predictions and confidence
        confidence, predictions = ([],[])
        for i, row in X_test.iterrows(): 
            center = kmeans.predict(row.reshape(1,-1))
            predictions.append(models[center[0]].predict(row.reshape(1,-1)))
            confidence.append(models[center[0]].predict_proba(row.reshape(1,-1)))
        return [c[0][-1] for c in confidence], [p[0] for p in predictions]
 
    def json_encode(self):
        base = tl_alg.Base_Transfer.json_encode(self)
        base.update({"num_clusters":self.num_clusters})
        return base
