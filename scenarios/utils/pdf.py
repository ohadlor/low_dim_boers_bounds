from __future__ import annotations
from typing import Optional, Self

import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gamma, factorial2, erf

from slicesInference.distributions import PDF

from .functions import distance


class TruncatedSphereMultiVariateGaussianPDF(PDF):
    """
    A class representing a truncated multivariate Gaussian probability density function (PDF).
    The distribution is assumed to have a covariance of std**2 * I, where I is the identity matrix.

    Parameters
    ----------
    mean : np.ndarray
        The mean vector of the distribution.
    std : float
        The standard deviation of the distribution.
    range : float
        The range of the distribution.
    rng : np.random.Generator
        The random number generator.
    """

    def __init__(
        self,
        dim: int,
        mean: np.ndarray,
        std: float,
        range: float,
        rng: np.random.Generator,
    ) -> None:
        # TODO support range = np.inf, regular multivariate normal distribution
        assert dim == len(mean), "mean and dim do not match"
        self.dim = dim
        self.mean = mean
        self.std = std
        self.cov = std**2 * np.eye(self.dim)
        self.range = range
        self.rng = rng

    @property
    def Normalizer(self):
        """
        Calculate the normalization factor for the given range and standard deviation.

        Returns
        -------
        float
            The normalization factor.
        """
        # TODO implement the normalization factor for the truncated sphere for higher dimensions.
        if self.dim != 2:
            raise NotImplementedError
            if self.dim // 2 == 0:
                integral = (
                    -multivariate_normal.pdf(self.range)
                    * factorial2(self.dim - 1)
                    * np.sum([self.range ** (2 * i) / factorial2(2 * i) for i in range(self.dim)])
                )
            else:
                integral = -multivariate_normal.pdf(self.range) * factorial2(self.dim - 1) * np.sum(
                    [self.range ** (2 * i) / factorial2(2 * i) for i in range(self.dim)]
                ) + factorial2(self.dim) * erf(self.range)
            unit_sphere_surface = 2 * np.pi ** (self.dim / 2) / gamma(self.dim / 2 + 1)
            return integral * unit_sphere_surface

        return 1 - np.exp(-1 / 2 * (self.range / self.std) ** 2)

    @property
    def Mu(self):
        """
        Get the mean vector of the PDF.

        Returns
        -------
        np.ndarray
            The mean vector.
        """
        return self.mean.reshape(-1, 1)

    @property
    def Min(self):
        """
        Get the minimum value of the PDF.

        Returns
        -------
        float
            The minimum value of the PDF.
        """
        return float(self.evaluate(self.mean + np.array([0, self.range])))

    @property
    def Max(self):
        """
        Get the maximum value of the PDF.

        Returns
        -------
        float
            The maximum value of the PDF.
        """
        return float(self.evaluate(self.mean))

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples from the truncated gaussian.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        np.ndarray
            The generated samples.
        """
        samples = []
        while n_samples > 0:
            new_samples = self.rng.multivariate_normal(self.mean, self.cov, n_samples)
            new_samples = new_samples[self._in_domain(new_samples)]
            n_samples = n_samples - len(new_samples)
            samples += new_samples.tolist()

        return np.asarray(samples)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the PDF at the given points.

        Parameters
        ----------
        points : np.ndarray
            The points at which to evaluate the PDF.

        Returns
        -------
        float
            The PDF values at the given points.
        """
        likelihood = np.array(multivariate_normal.pdf(points, mean=self.mean, cov=self.cov) / self.Normalizer).reshape(
            -1, 1
        )
        likelihood[np.logical_not(self._in_domain(points))] = 0
        return likelihood.flatten()

    def log_likelihood(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the log likelihood of the PDF at the given points.

        Parameters
        ----------
        points : np.ndarray
            The points at which to evaluate the PDF.

        Returns
        -------
        float
            The log likelihood of the PDF at the given points.
        """
        log_likelihood = multivariate_normal.logpdf(points, mean=self.mean, cov=self.cov) - np.log(self.Normalizer)
        log_likelihood[np.logical_not(self._in_domain(points))] = -np.inf
        return log_likelihood

    def copy(self) -> TruncatedSphereMultiVariateGaussianPDF:
        """
        Create a copy of the TruncatedMultiVariateGaussianPDF object.

        Returns
        -------
        TruncatedMultiVariateGaussianPDF
            The copied TruncatedMultiVariateGaussianPDF object.
        """
        return TruncatedSphereMultiVariateGaussianPDF(self.dim, self.mean.copy(), self.std, self.range, self.rng)

    def _in_domain(self, points: np.ndarray, e_tol: float = 1e-8) -> np.ndarray:
        """
        Check if the given points are in the domain of the PDF.

        Parameters
        ----------
        points : np.ndarray
            The points to check.
        e_tol : float
            The error tolerance for the distance.
        Returns
        -------
        np.ndarray
            A boolean array indicating if the points are in the domain of the PDF.
        """
        return np.array(distance(points, self.mean, axis=1) <= self.range + e_tol)

    def multiply(self, pdf):
        raise NotImplementedError


class Truncated2DGaussianPDF(TruncatedSphereMultiVariateGaussianPDF):
    """
    A class representing a truncated multivariate Gaussian probability density function (PDF) in 2D.
    The distribution is assumed to have a covariance of std**2 * I, where I is the identity matrix.

    Parameters
    ----------
    mean : np.ndarray
        The mean vector of the distribution.
    std : float
        The standard deviation of the distribution.
    range : float
        The range of the distribution.
    rng : np.random.Generator
        The random number generator.
    """

    def __init__(self, mean: np.ndarray, std: float, range: float, rng: np.random.Generator) -> None:
        # TODO support range = np.inf, regular multivariate normal distribution
        dim = 2
        super().__init__(dim, mean, std, range, rng)

    def copy(self) -> Truncated2DGaussianPDF:
        """
        Create a copy of the Truncated2DGaussianPDF object.

        Returns
        -------
        Truncated2DGaussianPDF
            The copied Truncated2DGaussianPDF object.
        """
        return Truncated2DGaussianPDF(self.mean.copy(), self.std, self.range, self.rng)


class MixturePDF(PDF):
    def __init__(
        self,
        mixture: list[PDF],
        weights: Optional[np.ndarray] = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        if weights is not None:
            assert len(mixture) == len(weights), "mixture and weights lengths do not match"
            # Normalize weights
            self.weights = np.array(weights) / np.sum(weights)
        else:
            self.weights = np.repeat([1 / len(mixture)], len(mixture))
        self.mixture = mixture
        self.rng = rng

    def re_weight(self, weights: np.ndarray):
        assert weights.shape == self.weights.shape, "weights dimensions error"

        self.weights = weights / np.sum(weights)

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sampling is done by first sampling a component from the mixture and
        then by sampling from that specific component
        """

        sample_components: list[PDF] = self.rng.choice(
            self.mixture,
            n_samples,
            p=self.weights,
        )

        samples = [component.sample(1) for component in sample_components]

        return np.asarray(samples)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        likelihoods = self.likelihood_matrix(points)

        return self.weights @ np.asarray(likelihoods)

    def likelihood_matrix(self, points: np.ndarray) -> np.ndarray:
        return np.asarray([component.evaluate(points) for component in self.mixture])

    def copy(self) -> Self:
        mixture = [x.copy() for x in self.mixture]
        return type(self)(mixture, self.weights.copy(), self.rng)


# Not needed
class Truncated2DGaussianMixturePDF(MixturePDF):
    def __init__(
        self,
        mixture: list[Truncated2DGaussianPDF],
        weights: Optional[np.ndarray] = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(mixture, weights, rng)

    def multiply(self, pdf):
        raise NotImplementedError
