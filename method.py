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

    def fit(self, X_train, y_train):
        # Save X and y
        self.X_train = X_train
        self.y_train = y_train

        # Firstly we analyze the training set to find majority class and to
        # establish the imbalance ratio
        self.classes, c_counts = np.unique(y_train, return_counts=True)
        majority_c = 0 if c_counts[0] > c_counts[1] else 1
        minority_c = 1 - majority_c

        min_idx = np.where(y_train == minority_c)[0]
        maj_idx = np.where(y_train == majority_c)[0]

        # K is the imbalanced ratio round to int, being also a number of
        # ensemble members.
        imbalance_ratio = c_counts[majority_c] / c_counts[minority_c]
        self.k = int(np.around(imbalance_ratio))
        self.k = self.k if self.k > 2 else 2

        # We use k to KFold division of majority class
        self.clfs = []
        kf = model_selection.KFold(n_splits=self.k, shuffle=True)
        for _, index in kf.split(maj_idx):
            fold_idx = np.concatenate([min_idx, maj_idx[index]])
            X_train_f, y_train_f = X_train[fold_idx], y_train[fold_idx]

            clf = base.clone(self.base_clf)
            clf.fit(X_train_f, y_train_f)
            self.clfs.append(clf)

        # Add OS
        clf = base.clone(self.base_clf)
        os = over_sampling.RandomOverSampler()
        X_os, y_os = os.fit_sample(self.X_train, self.y_train)
        clf.fit(X_os, y_os)
        self.clfs.append(clf)

        # Calculate weights as balanced accuracy on whole set
        self.weights = np.array(
            [metrics.balanced_accuracy_score(self.y_train,
                                             clf.predict(self.X_train))
             for clf in self.clfs]
        )

        scaler = preprocessing.MinMaxScaler()
        self.nweights = scaler.fit_transform(self.weights.reshape(-1,1)).T[0]
        self.nweights += .01


    def test(self, X_test):
        self.X_test = X_test
        self.esc = ensemble_support_cube(self.clfs, X_test)

        # Analyze dependencies
        segments = list(range(len(self.clfs)))
        p_treshold = 0.01
        for i in range(self.k):
            a = self.esc[i,:,0]
            for j in range(i+1,self.k):
                b = self.esc[j,:,0]
                p = stats.wilcoxon(a,b).pvalue
                c = (i,j)
                if p < p_treshold:
                    segments[i] = j

        # Reduced ESC and weights
        self.resc = np.array(
            [np.mean(self.esc[segments == i,:,:] * self.weights[segments == i, np.newaxis, np.newaxis], axis=0)
             for i in np.unique(segments)]
        )
        self.rweights = np.array(
            [np.mean(self.weights[segments == i])
             for i in np.unique(segments)]
        )

        scaler = preprocessing.MinMaxScaler()
        self.nrweights = scaler.fit_transform(self.rweights.reshape(-1,1)).T[0]
        self.nrweights += .01

        # Contrasts
        self.contrast = np.abs(self.esc[:,:,0] - self.esc[:,:,1])
        self.rcontrast = np.abs(self.resc[:,:,0] - self.resc[:,:,1])

        # Calculate all the measures
        self.wesc = self.esc * self.weights[:,np.newaxis,np.newaxis]
        self.nwesc = self.esc * self.nweights[:,np.newaxis,np.newaxis]
        self.cwesc = self.wesc * self.contrast[:,:,np.newaxis]
        self.ncwesc = self.nwesc * self.contrast[:,:,np.newaxis]

        self.rwesc = self.resc * self.rweights[:,np.newaxis,np.newaxis]
        self.nrwesc = self.resc * self.nrweights[:,np.newaxis,np.newaxis]
        self.rcwesc = self.rwesc * self.rcontrast[:,:,np.newaxis]
        self.nrcwesc = self.nrwesc * self.rcontrast[:,:,np.newaxis]

    def decision_cube(self, reduced = False, mode = 'regular', os = False):
        esc_ = None
        if reduced:
            if mode == 'regular':
                esc_ = self.esc
            elif mode == 'weighted':
                esc_ = self.wesc
            elif mode == 'cweighted':
                esc_ = self.cwesc
            elif mode == 'nweighted':
                esc_ = self.nwesc
            elif mode == 'ncweighted':
                esc_ = self.ncwesc
        else:
            if mode == 'regular':
                esc_ = self.resc
            elif mode == 'weighted':
                esc_ = self.rwesc
            elif mode == 'cweighted':
                esc_ = self.rcwesc
            elif mode == 'nweighted':
                esc_ = self.nrwesc
            elif mode == 'ncweighted':
                esc_ = self.nrcwesc

        if os:
            return esc_
        else:
            return esc_[:-1,:,:]

    def predict(self, esc):
        # Basic soft voting
        return np.argmax(np.sum(esc,axis=0), axis=1)


def ensemble_support_cube(clfs, X):
    return np.array([clf.predict_proba(X) for clf in clfs])

def regular_bac(base_clf, X_train, y_train, X_test, y_test):
    regular_clf = base.clone(base_clf)
    regular_clf.fit(X_train, y_train)
    regular_pred = regular_clf.predict(X_test)
    return metrics.balanced_accuracy_score(y_test,regular_pred)

def us_os_bac(base_clf, X_train, y_train, X_test, y_test):
    us = under_sampling.RandomUnderSampler()
    os = over_sampling.RandomOverSampler()
    X_us, y_us = us.fit_sample(X_train, y_train)
    X_os, y_os = os.fit_sample(X_train, y_train)
    us_clf = base.clone(base_clf)
    os_clf = base.clone(base_clf)
    us_clf.fit(X_us, y_us)
    os_clf.fit(X_os, y_os)
    us_pred = us_clf.predict(X_test)
    os_pred = os_clf.predict(X_test)
    return (
        metrics.balanced_accuracy_score(y_test,us_pred),
        metrics.balanced_accuracy_score(y_test,os_pred)
    )
