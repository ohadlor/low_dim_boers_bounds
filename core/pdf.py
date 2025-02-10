from __future__ import annotations
from warnings import warn
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gamma, factorial2, erf

from .utilities import distance


class PDF(ABC):
    """
    Abstract interface for a probability distribution function
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def sample(self, n_samples: int) -> np.ndarray:
        """
        returns array of samples where each row is a sample
        """
        pass

    @abstractmethod
    def likelihood(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def copy(self) -> PDF:
        pass

    @abstractmethod
    def _in_domain(self, points: np.ndarray) -> np.ndarray:
        pass


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
        rng: Optional[np.random.Generator | int] = None,
    ) -> None:
        # TODO support range = np.inf, regular multivariate normal distribution
        assert dim == len(mean), "mean and dim do not match"
        self.dim = dim
        self.mean = mean
        self.std = std
        self.cov = std**2 * np.eye(self.dim)
        self.range = range
        self.rng = np.random.default_rng(rng)

        if self.Normalizer == 1:
            warn("PDF is not truncated", UserWarning)

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
        return float(self.likelihood(self.mean + np.array([0, self.range])))

    @property
    def Max(self):
        """
        Get the maximum value of the PDF.

        Returns
        -------
        float
            The maximum value of the PDF.
        """
        return float(self.likelihood(self.mean))

    def sample(self, n_samples: int = 1) -> np.ndarray:
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
        samples = np.empty((n_samples, self.dim), dtype=float)
        while n_samples > 0:
            new_samples = self.rng.multivariate_normal(self.mean, self.cov, n_samples)
            new_samples = new_samples[self._in_domain(new_samples)]
            n_new_samples = len(new_samples)
            n_samples = n_samples - n_new_samples
            slicer = slice(-n_samples, -n_samples + n_new_samples)
            samples[slicer] = new_samples

        return samples

    def likelihood(self, points: np.ndarray) -> np.ndarray:
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
        shape = points.shape[:-1]
        likelihood = np.asarray(
            multivariate_normal.pdf(points, mean=self.mean, cov=self.cov) / self.Normalizer
        ).reshape(shape)
        ood_samples = np.logical_not(self._in_domain(points))
        likelihood[ood_samples] = np.zeros_like(likelihood[ood_samples])
        if np.any(ood_samples):
            warn("Some samples are out of domain")
        return likelihood

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

    def _in_domain(self, points: np.ndarray, e_tol: float = 1e-6) -> np.ndarray[bool]:
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
        shape = points.shape[:-1]
        return np.array(distance(points, self.mean, axis=-1) <= self.range + e_tol).reshape(shape)


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

    def __init__(
        self, mean: np.ndarray, std: float, range: float, rng: Optional[np.random.Generator | int] = None
    ) -> None:
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


class MultiVariateGaussianPDF(PDF):

    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        Lambda: np.ndarray = None,
        rng: Optional[np.random.Generator | int] = None,
    ):
        """
        mean: 1-D array_like, of length N
              Mean of the N-dimensional distribution.

        cov: 2-D array_like, of shape (N, N)
             Covariance matrix of the distribution.
             It must be symmetric and positive-semidefinite
             for proper sampling
        """
        assert (
            mean.shape[0],
            mean.shape[0],
        ) == cov.shape, "mean and cov dimensions error"

        self.mean = mean
        self.cov = cov
        if Lambda is not None:
            self.__lambda = Lambda
        else:
            self.__lambda = np.linalg.inv(self.cov)

        self.rng = np.random.default_rng(rng)

    @property
    def Lambda(self):
        return self.__lambda

    @property
    def Mu(self):
        return self.mean.reshape(-1, 1)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return self.rng.multivariate_normal(self.mean, self.cov, n_samples)

    def likelihood(self, points: np.ndarray):
        return multivariate_normal.pdf(points, mean=self.mean, cov=self.cov)

    def copy(self) -> MultiVariateGaussianPDF:
        return MultiVariateGaussianPDF(self.mean.copy(), self.cov.copy(), rng=self.rng)

    def _in_domain(self, points: np.ndarray) -> bool:
        # infinite support
        return True
