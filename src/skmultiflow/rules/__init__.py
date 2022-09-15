"""
The :mod:`skmultiflow.rules` module includes rule-based learning methods.
"""

from .amrules_classifier import AMRulesClassifier
from .amrules_regressor import AMRulesRegressor
from .very_fast_decision_rules import VeryFastDecisionRulesClassifier
from .very_fast_decision_rules import VFDR   # remove in v0.7.0

__all__ = ["AMRulesClassifier", "AMRulesRegressor", "VeryFastDecisionRulesClassifier", "VFDR"]
