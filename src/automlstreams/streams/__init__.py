"""
The :mod:`automlstreams.streams` module contains streams implementations for the
`skmultiflow.data.base_stream` abstract class
"""

from .kafka_stream import KafkaStream

__all__ = ['KafkaStream']
