from __future__ import annotations
from typing import Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .landmark import Landmark
    from .environment import Environment
    from .pdf import PDF


class ParticleBelief:
    # Number of particles in the full belief
    full_n: int = 100

    def __init__(self, particles: np.ndarray, weights: np.ndarray, rng: Optional[np.random.Generator | int] = None):
        """
        Initialize the ParticleBelief with particles, weights, and an optional random number generator.

        Parameters
        ----------
        particles : np.ndarray
            Array of particles representing the belief.
        weights : np.ndarray
            Array of weights corresponding to each particle.
        rng : Optional[np.random.Generator | int], optional
            Random number generator or seed for reproducibility, by default None.
        """
        self.particles = particles
        self.weights = weights
        self.rng = np.random.default_rng(rng)

    def prediction_step(self, transition: np.ndarray, transition_noise: PDF):
        """
        Perform the prediction step of the particle filter.

        Parameters
        ----------
        transition : np.ndarray
            Array representing the transition to be applied to each particle.
        transition_noise : PDF
            Probability density function representing the transition noise.
        """
        self.particles += transition + transition_noise.sample(self.n_particles)
        self.validate_weights()

    def filter_step(self, da_space: np.ndarray, environment: Environment):
        """
        Perform the filtering step of the particle filter.

        Parameters
        ----------
        da_space : np.ndarray
            Array representing the data association space.
        environment : Environment
            The environment in which the particles are being filtered.
        """
        # Remove particles that are not in the data association space
        filtered_indicies = environment.in_da_space(self.particles, da_space)
        if filtered_indicies.all():
            return
        self.particles = self.particles[filtered_indicies]
        self.weights = self.weights[filtered_indicies]
        self.normalize()
        self.validate_weights()

    def update_step(
        self,
        landmarks: Sequence[Landmark],
        measurements: Sequence[np.ndarray],
        observation_noise: PDF,
    ):
        """
        Perform the update step of the particle filter.

        Parameters
        ----------
        landmarks : Sequence[Landmark]
            List of landmarks in the environment.
        measurements : Sequence[np.ndarray]
            List of measurements corresponding to each landmark.
        observation_noise : PDF
            Probability density function representing the observation noise.
        """
        # TODO: Resample for the updated belief, should not affect reward, check
        if self.n_particles != self.full_n:
            self.resample()
        for landmark, measurement in zip(landmarks, measurements):
            noise = measurement - (self.particles - landmark.loc)
            likelihood = observation_noise.likelihood(noise)
            self.weights *= likelihood
        # Remove zero likelihood particles, should not happen
        if np.any(self.weights == 0):
            zero_indices = self.weights == 0
            self.particles = self.particles[~zero_indices]
            self.weights = self.weights[~zero_indices]
            self.normalize()
            self.resample()
        else:
            self.normalize()
        self.validate_weights()

    def filter_observation_space(
        self, landmarks: Sequence[Landmark], measurements: Sequence[np.ndarray], observation_noise: PDF
    ):
        """
        Remove particles that are out of domain of the full observation.

        Parameters
        ----------
        landmarks : Sequence[Landmark]
            List of landmarks in the environment.
        measurements : Sequence[np.ndarray]
            List of measurements corresponding to each landmark.
        observation_noise : PDF
            Probability density function representing the observation noise.
        """
        in_domain = np.ones(self.n_particles, dtype=bool)
        for landmark, measurement in zip(landmarks, measurements):
            noise = measurement - (self.particles - landmark.loc)
            in_domain = np.logical_and(observation_noise._in_domain(noise), in_domain)
        self.particles = self.particles[in_domain]
        self.weights = self.weights[in_domain]
        self.normalize()
        self.validate_weights()

    def normalize(self):
        self.weights /= self.weights.sum()

    def resample(self, n_samples: Optional[int] = None):
        if n_samples is None:
            n_samples = self.full_n
        self.particles = self.sample(n_samples)
        self.weights = np.ones(n_samples) / n_samples

    def sample(self, n_samples: int) -> np.ndarray:
        return self.rng.choice(self.particles, n_samples, p=self.weights)

    def copy(self) -> ParticleBelief:
        return ParticleBelief(self.particles.copy(), self.weights.copy(), self.rng)

    @property
    def n_particles(self) -> int:
        return len(self.particles)

    def validate_weights(self):
        if np.isnan(self.weights).any():
            raise ValueError("Weights contain NaN values.")
        if (self.weights == 0).any():
            raise ValueError("Weights contain zero values.")
        if len(self.weights) == 0:
            raise ValueError("Weights are empty.")
