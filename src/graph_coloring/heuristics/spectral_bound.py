from typing import Optional

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
from math import ceil

from graph_coloring.dynamics.base_class import Coloring
from graph_coloring.heuristics.base import DualBound


class SpectralBound(DualBound):
    def _compute_reward_to_go(self, partial_coloring: Optional = None):
        if partial_coloring is None:
            partial_coloring = Coloring(self.graph)
    
        laplacian = partial_coloring.complement_graph.laplacian
        spectrum_bounds = SpectralBound.spectrum_bounds(laplacian)
    
        return ceil(max(spectrum_bounds))
    
    @staticmethod
    def spectrum_bounds(laplacian: csr_matrix):
        L = laplacian
        A = diags(L.diagonal()) - L
        Q = L + 2 * A
        
        mu, _ = eigsh(A.asfptype())
        mu = sorted(list(mu), reverse=True)
        
        theta, _ = eigsh(L.asfptype())
        theta = sorted(list(theta), reverse=True)
        
        delta, _ = eigsh(Q.asfptype())
        delta = sorted(list(delta), reverse=True)
        
        bounds = [
            1 + mu[0] / (-mu[-1]),  # Adjacency bound
            1 + mu[0] / (theta[0] - mu[0]),  # (2)
            1 + mu[0] / (mu[0] - delta[0] + theta[0]),  # (3)
            1 + mu[0] / (mu[0] - delta[-1] + theta[-1]),  # (4)
        ]
        
        n = len(mu)
        for m in range(1, n + 1):
            bounds.append(1 + sum(mu[i] for i in range(m)) / -sum(mu[n - i - 1] for i in range(m)))  # (5)
            bounds.append(1 + sum(mu[i] for i in range(m)) / -sum(theta[i] - mu[i] for i in range(m)))  # (6)
            bounds.append(1 + sum(mu[i] for i in range(m)) / -sum(mu[i] - delta[i] + theta[i] for i in range(m)))  # (7)
            bounds.append(1 + sum(mu[i] for i in range(m)) / -sum(
                mu[i] - delta[n - i - 1] + theta[n - i - 1] for i in range(m)
            ))  # (8)
        
        return bounds
