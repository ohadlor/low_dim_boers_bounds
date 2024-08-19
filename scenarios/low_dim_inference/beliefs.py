from __future__ import annotations
from typing import Sequence

import numpy as np

from slicesInference.distributions.pdf import PDF

from scenarios.utils.beliefs import Belief


class ParticleBelief(Belief):
    def __init__(self, particles: Sequence[np.ndarray], weights: Sequence[float], rng: np.random.Generator) -> None:
        """
        Initializes an instance of MyClass.

        Args:
            slices_factor_graph (SlicesFactorGraph): The slices factor graph object.

        """
        self.current_particles = particles
        assert abs(sum(weights) - 1) < 1e-6
        self.current_weights = weights
        self.N = len(particles)

        self.rng = rng

    def propogate(
        self,
        transition: float | np.ndarray,
        noise: PDF,
    ) -> None:
        """propogate particles

        Parameters
        ----------
        transition : float | np.ndarray | Function
            recieved action
        noise : PDF
            noise of factor
        """
        samples = noise.sample(len(self.current_particles))
        self.next_particles = transition + samples + self.current_particles

    def update(
        self,
        noise: PDF,
        remove_zero_weights: bool = True,
    ) -> None:
        """Update weights via observation

        Parameters
        ----------
        measurements: Sequence[float] | Sequence[np.ndarray] | Sequence[Function]
            recieved measurement
        noises : PDF
            factor noise
        """

        likelihood = noise.evaluate(self.next_particles)
        self.next_weights = self.normalize_weights(self.current_weights * likelihood)
        if remove_zero_weights:
            mask = self.next_weights > 0
            self.current_weights = self.normalize_weights(self.current_weights[mask])
            self.current_particles = self.current_particles[mask]
            self.next_weights = self.normalize_weights(self.next_weights[mask])
            self.next_particles = self.next_particles[mask]
            self.N = len(self.current_particles)

    @staticmethod
    def normalize_weights(weights: Sequence[float]) -> np.ndarray[float]:
        return weights / np.sum(weights)

    def resample(self, N: int) -> None:
        """Resample particles"""
        self.current_particles = self.rng.choice(self.current_particles, size=N, p=self.current_weights)
        self.current_weights = np.ones(N) / N

    def sample(self, n: int) -> set[tuple[float, np.ndarray]]:
        """Sample particles"""
        return self.rng.choice(self.next_particles, size=n, p=self.next_weights)

    def subset(self, indicies: np.ndarray) -> tuple[np.ndarray[float], np.ndarray, np.ndarray[float], np.ndarray]:
        """Subset particles"""
        return (
            self.current_weights[indicies],
            self.current_particles[indicies],
            self.next_weights[indicies],
            self.next_particles[indicies],
        )

    def size_to_indicies(self, n: int | float) -> np.ndarray:
        """Subset particles"""
        if isinstance(n, float):
            n = int(n * self.N)
        assert n <= self.N
        return self.rng.integers(self.N, size=n)

    def copy(self) -> ParticleBelief:
        """Copy belief as next belief"""
        return ParticleBelief(self.next_particles, self.next_weights, self.rng)

    def inference(self) -> None:
        """Not used"""
        pass

    def prediction_step(self) -> None:
        pass

    def update_step(self) -> None:
        pass
