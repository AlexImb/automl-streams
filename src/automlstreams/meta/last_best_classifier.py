import numpy as np
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.trees import HoeffdingTree as HT
from sklearn.naive_bayes import MultinomialNB as MNB


class LastBestClassifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Classifier that keeps a set of base estimators in a leaderboard
    and pick the estimator for the next window best on the prediction
    accuracy of the estimator in the previous window.

    Parameters
    ----------
    estimators: list of skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator
        default=[DecisionTreeClassifier()]
        A list of estimators for the leaderboard

    window_size: int (default=100)
        The size of the window used for extracting meta-features.

    active_learning: boolean (default=True)
        Switches between using the fit() or partial_fit() method of the base estimators

    Notes
    -----


    """

    def __init__(self,
                 estimators=[MNB(), HT()],
                 window_size=100,
                 active_learning=True):

        self.estimators = estimators
        self.leader_index = 0
        self.window_size = window_size
        self.active_learning = active_learning
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
                # Train base estimators in a prequential way
                if self.w > 0:
                    self.leader_index = self._get_leader_base_estimator_index(X, y)

                self._partial_fit_estimators(X, y, classes)
                self.w += 1
                self.i = -1

        return self

    def _partial_fit_estimators(self, X, y, classes, sample_weight=None):
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
        for index, base_estimator in enumerate(self.estimators):
            try:
                if self.active_learning is True:
                    self.estimators[index] = base_estimator.partial_fit(X, y, classes, sample_weight)
                else:
                    try:
                        self.estimators[index] = base_estimator.fit(X, y, classes, sample_weight)
                    except TypeError:
                        self.estimators[index] = base_estimator.fit(X, y, sample_weight=sample_weight)
            except AttributeError:
                try:
                    self.estimators[index] = base_estimator.fit(X, y, classes, sample_weight)
                except TypeError:
                    self.estimators[index] = base_estimator.fit(X, y, sample_weight=sample_weight)

    def _get_leader_base_estimator_index(self, X, y):
        try:
            scores = [be.score(X, y) for be in self.estimators]
        except NotImplementedError:
            from sklearn.metrics import accuracy_score
            scores = [accuracy_score(y, be.predict(X), normalize=False) for be in self.estimators]

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
        return self.estimators[self.leader_index].predict_proba(X)

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
        return self.estimators[self.leader_index].predict(X)

    def reset(self):
        """ Resets the estimator to its initial state.

        Returns
        -------
            self

        """
        self.estimators = [be.reset() for be in self.estimators]
        self.leader_index = 0
        self.w = 0
        self.i = -1
        self.X_window = None
        self.y_window = None
        return self
