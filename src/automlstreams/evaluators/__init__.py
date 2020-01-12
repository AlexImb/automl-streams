"""
The :mod:`automlstreams.evaluators` module contains evaluators implementations
for the `StreamEvaluator` abstract class from the :mod:`skmultiflow.evaluation`
module
"""

from .evaluate_pretrained import EvaluatePretrained

__all__ = ['EvaluatePretrained']
