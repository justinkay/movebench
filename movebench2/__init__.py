"""Next-generation MoveBench pipeline (work-in-progress).

This package is a greenfield implementation intended to provide a cleaner
architecture (layer registry + YAML configs + feature store).
"""

from movebench2.study import Study

__all__ = ["Study"]
