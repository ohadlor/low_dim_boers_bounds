import numpy as np

from slicesInference.factor_graph.factors import SymbolicR2PairWiseLinearFactor, PriorFactor, EuclideanGroup
from scenarios.utils import Truncated2DGaussianPDF


class Truncated2DGaussianPairWiseLinearFactor(SymbolicR2PairWiseLinearFactor):
    def __init__(self, measurement: np.ndarray, noise: Truncated2DGaussianPDF) -> None:
        # xi_symbol = 0, xj_symbol = 1
        super().__init__(0, 1, measurement, noise)


class Truncated2DGaussianPriorFactor(PriorFactor):
    def __init__(self, noise: Truncated2DGaussianPDF) -> None:
        # symbol: int = 1
        super().__init__(1, noise)

    def get_euclidean_group(self) -> EuclideanGroup:
        return EuclideanGroup.R2

    def copy(self):
        pass

    def multiply(self):
        pass

    def is_mixture(self) -> bool:
        return False

    def is_multiplicable(self) -> bool:
        return False
