"""
con_noccai_v7
=============
Inference package for the hierarchy_v7 NOC 2021 hierarchical classifier.

Architecture : DistilBERT-base-uncased (English) + 5-level classification heads
Training run : hierarchy_v7, 15 epochs, best val L5 accuracy 93.98 %

Public API
----------
    from con_noccai_v7 import NOCPredictor
    from con_noccai_v7 import LabelMap           # advanced use
    from con_noccai_v7 import HierarchicalNOCModel, load_checkpoint  # advanced use
"""

from .predictor import NOCPredictor
from .label_map  import LabelMap
from .model      import HierarchicalNOCModel, load_checkpoint

__all__ = [
    "NOCPredictor",
    "LabelMap",
    "HierarchicalNOCModel",
    "load_checkpoint",
]

__version__ = "7.0.0"
