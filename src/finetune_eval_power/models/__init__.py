"""Keras models for both arms, and the deterministic training loop around them.

Everything in this package imports TensorFlow, which is an optional extra. Nothing
that computes a published number imports this package. The split is enforced by a
CI job that installs without the extra and runs the full analysis path.
"""
