from imblearn import under_sampling, over_sampling
from sklearn import base, model_selection, metrics, preprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class UndersampledEnsemble():
    """
    Shuffled K-fold undersampled ensemble.
    """
    def __init__(self, base_clf, dbname):
        self.base_clf = base_clf
        self.dbname = dbname
        self.clfs = []

    def fit(self, X, y):
        #print("Fit")
        # Save X and y
        self.X = X
        self.y = y

        # Firstly we analyze the training set to find majority class and to
        # establish the imbalance ratio
        classes, c_counts = np.unique(y, return_counts=True)
        majority_c = 0 if c_counts[0] > c_counts[1] else 1
        minority_c = 1 - majority_c

        min_idx = np.where(y == minority_c)[0]
        maj_idx = np.where(y == majority_c)[0]

        imbalance_ratio = c_counts[majority_c] / c_counts[minority_c]

        # K is the imbalanced ratio round to int, being also a number of
        # ensemble members.
        self.k = int(imbalance_ratio)
        self.k = self.k if self.k > 2 else 2

        # We use k to KFold division of majority class
        self.clfs = []
        kf = model_selection.KFold(n_splits=self.k, shuffle=True)
        for _, index in kf.split(maj_idx):
            fold_idx = np.concatenate([min_idx, maj_idx[index]])
            X_train, y_train = X[fold_idx], y[fold_idx]

            clf = base.clone(self.base_clf)
            clf.fit(X_train, y_train)
            self.clfs.append(clf)

        # Add OS
        clf = base.clone(self.base_clf)
        os = over_sampling.RandomOverSampler()
        X_os, y_os = os.fit_sample(X, y)
        clf.fit(X_os, y_os)
        self.clfs.append(clf)


        # Calculate weights as balanced accuracy on whole set
        self.weights = np.array(
            [metrics.balanced_accuracy_score(y,clf.predict(X))
             for clf in self.clfs]
        )
        scaler = preprocessing.MinMaxScaler()
        #print(self.weights)
        self.weights = scaler.fit_transform(self.weights.reshape(-1,1)).T[0]
        #print(self.weights)
        #exit()
        self.weights += .01

    def test(self, X_test):
        self.X_test = X_test
        self.esc = ensemble_support_cube(self.clfs, X_test)

        # Analyze dependencies
        segments = list(range(self.k+1))
        p_treshold = 0.01
        for i in range(self.k):
            a = self.esc[i,:,0]
            for j in range(i+1,self.k):
                b = self.esc[j,:,0]
                p = stats.wilcoxon(a,b).pvalue
                c = (i,j)
                if p < p_treshold:
                    segments[i] = j

        self.nesc = np.array(
            [np.mean(self.esc[segments == i,:,:], axis=0)
             for i in np.unique(segments)]
        )
        self.nweights = np.array(
            [np.mean(self.weights[segments == i])
             for i in np.unique(segments)]
        )

        self.calculate()

    def reduce(self):
        self.esc = self.nesc
        self.weights = self.nweights

    def calculate(self):
        # Calculate all the measures
        self.wesc = self.esc * self.weights[:,np.newaxis,np.newaxis]
        self.contrast = np.abs(self.esc[:,:,0] - self.esc[:,:,1])
        self.cwesc = self.wesc * self.contrast[:,:,np.newaxis]




        #exit()

    def predict(self, X):
        # Basic soft voting
        return np.argmax(np.sum(self.esc,axis=0), axis=1)

    def predict_bac_weighted(self, X):
        # Weighting with bac
        return np.argmax(np.sum(self.wesc,axis=0), axis=1)

    def predict_c_weighted(self, X):
        # Contrasted bac
        return np.argmax(np.sum(self.cwesc,axis=0), axis=1)


def ensemble_support_cube(clfs, X):
    return np.array([clf.predict_proba(X) for clf in clfs])

class BalancingEnsemble():
    def __init__(self, base_clf):
        self.base_clf = base_clf
        self.pool = [
            under_sampling.CondensedNearestNeighbour(),
            under_sampling.EditedNearestNeighbours(),
            under_sampling.RepeatedEditedNearestNeighbours(),
            under_sampling.AllKNN(),
            under_sampling.InstanceHardnessThreshold(),
            under_sampling.NearMiss(),
            under_sampling.NeighbourhoodCleaningRule(),
            under_sampling.OneSidedSelection(),
            under_sampling.RandomUnderSampler(),
            under_sampling.TomekLinks(),
            over_sampling.ADASYN(n_neighbors=3),
            over_sampling.RandomOverSampler(),
            over_sampling.SMOTE(k_neighbors=3)
        ]
        self.clfs = []

    def fit(self, X, y):
        for estimator in self.pool:
            X_, y_ = estimator.fit_sample(X,y)
            clf = base.clone(self.base_clf)
            clf.fit(X_,y_)
            self.clfs.append(clf)

    def score(self, X,y):
        print("Member scores")
        for clf in self.clfs:
            print("%.3f" % clf.score(X,y))
        return 0

    def ensemble_predict_proba(self, X):
        a = [clf.predict_proba(X) for clf in self.clfs]
        return a
