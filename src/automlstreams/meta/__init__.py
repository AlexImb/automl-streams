"""
The :mod:`automlstreams.meta` module contains meta-learning algorithms
"""

from .meta_classifier import MetaClassifier
# TODO: Add MetaRegressor implementation
# from .meta_regressor import MetaRegressor
from .last_best_classifier import LastBestClassifier

__all__ = ['MetaClassifier', 'LastBestClassifier']
