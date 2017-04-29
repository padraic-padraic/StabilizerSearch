# coding: utf-8

"""The StabilizerSearch.search module contains a pair of base classes,
_Search and _Result, that are used to find and return the Stabilizer Rank
decomposition of a given state.

The other ssubmodules each define a strategy for finding the Stabilizer Rank.

Those methods are imported here to be made available to the rest of the package."""

from .brute_force import BruteForceSearch
from .random_walk import RandomWalkSearch
