from __future__ import annotations
from typing import Optional

import numpy as np


class Landmark:
    def __init__(
        self,
        location: np.ndarray,
        success_prob: float = 1.0,
        rng: Optional[np.random.Generator | int] = None,
    ) -> None:
        self.loc = location
        self.success_prob = success_prob
        self.rng = np.random.default_rng(rng)
        self.id = None
        """
        Initialize a Landmark object.

        Parameters
        ----------
        location : np.ndarray
            The location of the landmark.
        success_prob : float, optional
            The probability of success at the landmark (default is 1.0).
        rng : np.random.Generator or int or None, optional
            A random number generator instance, seed, or None to use the default RNG (default is None).
        """

    def success(self, n_samples: int = 1) -> np.ndarray[bool]:
        """Check if success occurs.

        Returns
        -------
        bool
            True if success occurs, False otherwise.
        """
        return self.rng.binomial(1, self.success_prob, n_samples).astype(bool)
