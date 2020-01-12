# -*- coding: utf-8 -*-

import pytest
from automlstreams.console import fib

__author__ = "Alexandru-Ionut Imbrea"
__copyright__ = "Alexandru-Ionut Imbrea"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
