import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.trees import HoeffdingTree as HT
from skmultiflow.bayes import NaiveBayes as NB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import SGDClassifier


from sklearn.model_selection import train_test_split


class MetaClassifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Meta Classifier that uses meta-learning for selecting the best
    base estimator for a certain window


    Parameters
    ----------
    meta_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator
        default=GradientBoostingRegressor
        Metalearner used to predict the best base estimator.

    base_estimators: list of skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator
        default=[DecisionTreeClassifier()]
        A list of base estimators.

    window_size: int (default=100)
        The size of the window used for extracting meta-features.

    Notes
    -----


    """

    def __init__(self,
                 meta_estimator=SGDClassifier(),
                 base_estimators=[MNB(), HT()],
                 mfe_groups=['general', 'statistical', 'info-theory'],
                 window_size=100):

        self.meta_estimator = meta_estimator
        self.base_estimators = base_estimators
        self.leader_index = 0
        self.mfe_groups = mfe_groups
        self.window_size = window_size
        self.w = 0
        self.i = -1

        self.X_window = None
        self.y_window = None

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. Usage varies depending on the learning method.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.
            Usage varies depending on the learning method.

        Returns
        -------
            self

        """
        r, c = X.shape

        if self.i < 0:
            self.X_window = np.zeros((self.window_size, c))
            self.y_window = np.zeros(self.window_size)
            self.i = 0

        for j in range(r):
            self.X_window[self.i] = X[j]
            self.y_window[self.i] = y[j]
            self.i += 1

            if self.i == self.window_size:
                # Extract meta-features
                mfe = MFE(self.mfe_groups, suppress_warnings=True).fit(self.X_window, self.y_window)
                metafeatures = np.array([mfe.extract()[1]])
                # metafeatures = pd.DataFrame(mfe.extract()[1])
                # metafeatures = metafeatures.dropna()
                # matefeatures = metafeatures.to_numpy()
                metafeatures[~np.isfinite(metafeatures)] = 0
                # metafeatures = metafeatures[~np.isnan(metafeatures)]

                # Select leader for predictions
                if self.w > 0:
                    predicted = self.meta_estimator.predict(metafeatures)
                    self.leader_index = predicted[0]
                    # print(f'Predicted leader for window {self.w}: {predicted}')

                # Train base estimators
                X_window_train, X_window_test, y_window_train, y_window_test = train_test_split(X, y)
                self._partial_fit_base_estimators(X_window_train, y_window_train, classes)
                leader_index = self._get_leader_base_estimator_index(X_window_test, y_window_test)
                # print(f'Actual leader for window {self.w}: {leader_index}')

                # Train meta learner
                metaclasses = [c for c in range(len(self.base_estimators))]
                self.meta_estimator.partial_fit(metafeatures, [leader_index], metaclasses)

                self.w += 1
                self.i = -1

        return self

    def _partial_fit_base_estimators(self, X, y, classes, sample_weight=None):
        """ Partially (incrementally) fit the base estimators.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model base estimators

        y: numpy.ndarray of shape (n_samples)
            An array-like with the labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. Usage varies depending on the learning method.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.
            Usage varies depending on the learning method.

        Returns
        -------
            self

        """
        for index, base_estimator in enumerate(self.base_estimators):
            try:
                self.base_estimators[index] = base_estimator.partial_fit(X, y, classes, sample_weight)
            except AttributeError:
                self.base_estimators[index] = base_estimator.fit(X, y, classes, sample_weight)

    def _get_leader_base_estimator_index(self, X, y):
        try:
            scores = [be.score(X, y) for be in self.base_estimators]
        except NotImplementedError:
            from sklearn.metrics import accuracy_score
            scores = [accuracy_score(y, be.predict(X), normalize=False) for be in self.base_estimators]

        return scores.index(max(scores))

    def predict_proba(self, X):
        """ Estimates the probability of each sample in X belonging to each of the class-labels.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the class probabilities for.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated with the X 
        entry of the same index. And where the list in index [i] contains len(self.target_values) elements,
        each of which represents the probability that the i-th sample of X belongs to a certain class-label.

        """
        return self.base_estimators[self.leader_index].predict(X)

    def predict(self, X):
        """ Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        return self.base_estimators[self.leader_index].predict(X)

    def reset(self):
        """ Resets the estimator to its initial state.

        Returns
        -------
            self

        """
        self.meta_estimator = self.meta_estimator.reset()
        self.base_estimators = [be.reset() for be in self.base_estimators]
        self.leader_index = 0
        self.w = 0
        self.i = -1
        self.X_window = None
        self.y_window = None
        return self
