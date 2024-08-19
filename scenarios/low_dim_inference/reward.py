from __future__ import annotations
import time

import numpy as np

from .beliefs import ParticleBelief
from .factors import Truncated2DGaussianPairWiseLinearFactor, Truncated2DGaussianPriorFactor


class LightDarkReward:
    def __init__(
        self,
        transition_noise: Truncated2DGaussianPairWiseLinearFactor,
        observation_noise: Truncated2DGaussianPriorFactor,
        belief: ParticleBelief,
        partial_simplification: float = 0.5,
        sith_simplification: float | None = None,
    ) -> None:

        self.belief = belief

        self.N = len(belief.current_particles)
        self.n_p = int(self.N * partial_simplification)
        self.p_indicies = self.belief.size_to_indicies(self.n_p)
        if sith_simplification == partial_simplification:
            self.n_s = self.n_p
            self.s_indicies = self.p_indicies
        elif sith_simplification is None:
            sith_simplification = partial_simplification**2
            self.n_s = int(self.N * sith_simplification)
            self.s_indicies = self.belief.size_to_indicies(self.n_s)
        else:
            self.n_s = int(self.N * sith_simplification)
            self.s_indicies = self.belief.size_to_indicies(self.n_s)

        self.obs_noise = observation_noise
        self.action_noise = transition_noise
        self.Mx = self.action_noise.noise.Max
        self.mx = self.action_noise.noise.Min
        self.Mz = self.obs_noise.noise.Max
        self.mz = self.obs_noise.noise.Min

    def boers_estimator(self) -> tuple[float, float]:
        t_0 = time.perf_counter()
        prior_particles = self.belief.current_particles
        prior_weights = self.belief.current_weights
        particles = self.belief.next_particles
        weights = self.belief.next_weights

        obs_noise_eval = self.obs_likelihood(particles)
        if np.any(np.where(weights == 0)):
            raise ValueError("Weights cannot be zero")
        if np.any(np.where(obs_noise_eval == 0)):
            raise ValueError("Observation likelihood cannot be zero")
        action_noise_eval = self.trans_likelihood(prior_particles, particles)

        reward = np.log(np.sum(prior_weights * obs_noise_eval)) - np.sum(
            weights * np.log(obs_noise_eval * np.sum(prior_weights * action_noise_eval, axis=1))
        )

        t_1 = time.perf_counter()
        delta_t = t_1 - t_0
        return reward, delta_t

    def partial_expectation_bounds(self) -> tuple[tuple[float, float], float]:
        t_0 = time.perf_counter()
        prior_weights, prior_particles, weights, particles = self.belief.subset(self.p_indicies)

        obs_noise_eval = self.obs_likelihood(particles)
        action_noise_eval = self.trans_likelihood(prior_particles, particles)

        sum_prior_weights = np.sum(prior_weights)
        sum_weights = np.sum(weights)

        simplified_estimator = -np.sum(weights * np.log(obs_noise_eval)) - (1 - sum_weights) * np.log(
            1 + sum_prior_weights
        )
        lower_bound = (
            np.log(np.sum(prior_weights * obs_noise_eval) + (1 - sum_prior_weights) * self.mz)
            - np.sum(weights * np.log(np.sum(prior_weights * action_noise_eval, axis=1) + self.Mx))
            - (1 - sum_weights) * (np.log(self.Mx) + np.log(self.Mz))
        )
        upper_bound = (
            np.log(np.sum(prior_weights * obs_noise_eval) + (1 - sum_prior_weights) * self.Mz)
            - np.sum(weights * np.log(np.sum(prior_weights * action_noise_eval, axis=1) + self.mx))
            - (1 - sum_weights) * (np.log(self.mx) + np.log(self.mz))
        )

        t_1 = time.perf_counter()
        delta_t = t_1 - t_0

        return (simplified_estimator + lower_bound, simplified_estimator + upper_bound), delta_t

    def sith_bounds(self) -> tuple[tuple[float, float], float]:
        t_0 = time.perf_counter()
        prior_particles = self.belief.current_particles
        prior_weights = self.belief.current_weights
        particles = self.belief.next_particles
        weights = self.belief.next_weights

        simplified_prior_weights, simplified_prior_particles, simplified_weights, simplified_particles = (
            self.belief.subset(self.s_indicies)
        )

        comp_indicies = np.setdiff1d(np.arange(self.N), self.s_indicies)
        comp_particles = self.belief.next_particles[comp_indicies]
        comp_weights = self.belief.next_weights[comp_indicies]

        simplified_obs_noise_eval = self.obs_likelihood(simplified_particles)

        simplified_obs_sum = np.sum(simplified_prior_weights * simplified_obs_noise_eval)

        a_lower = np.log(simplified_obs_sum)
        a_upper = np.log(simplified_obs_sum + self.Mz * (1 - np.sum(simplified_prior_weights)))
        b_lower = -np.sum(comp_weights * np.log(self.Mx * self.obs_likelihood(comp_particles))) - np.sum(
            simplified_weights
            * np.log(
                simplified_obs_noise_eval
                * np.sum(
                    prior_weights * self.trans_likelihood(prior_particles, simplified_particles),
                    axis=1,
                )
            )
        )
        b_upper = -np.sum(
            weights
            * np.log(
                self.obs_likelihood(particles)
                * np.sum(
                    simplified_prior_weights * self.trans_likelihood(simplified_prior_particles, particles),
                    axis=1,
                )
            )
        )

        t_1 = time.perf_counter()
        delta_t = t_1 - t_0

        return (a_lower + b_lower, a_upper + b_upper), delta_t

    def obs_likelihood(self, particles: np.ndarray) -> np.ndarray:
        return self.obs_noise.likelihood(particles)

    def trans_likelihood(self, prior_particles: np.ndarray, particles: np.ndarray) -> np.ndarray:
        """get transition likelihood of particles

        Parameters
        ----------
        prior_particles : np.ndarray
            _description_
        particles : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            return transition likelihood matrix where ith COLUMN is the likelihood of ith prior particle transitioning
            to all particles
        """
        return self.action_noise.likelihood(
            np.array(
                [np.repeat(prior_particles, len(particles), axis=0), np.tile(particles, (len(prior_particles), 1))]
            )
        ).reshape((len(particles), len(prior_particles)), order="F")
