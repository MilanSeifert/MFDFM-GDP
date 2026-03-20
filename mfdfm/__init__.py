"""Mixed-Frequency Dynamic Factor Model (Galli, 2017).

Produces a monthly business cycle index by combining monthly and quarterly
indicators via a state-space DFM estimated with the Kalman smoother.
"""

from mfdfm.model import MFDFM

__all__ = ["MFDFM"]
