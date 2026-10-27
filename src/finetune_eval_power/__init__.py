"""Measures how small a regression a fixed eval suite could have detected.

The package exists because a passing eval suite is routinely read as evidence that
fine-tuning broke nothing, and it is not. A suite can only see a regression large
enough to move the cases it happens to contain. This package computes that floor,
then goes looking for capability regions that fall below it.
"""

__version__ = "0.1.0"
