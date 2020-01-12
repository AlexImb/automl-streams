# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound
import logging

_logger = logging.getLogger(__name__)

try:
    dist_name = 'automl-streams'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound
