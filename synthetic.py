import logging
import igraph as ig
import networkx as nx
import numpy as np
from scipy.special import expit as sigmoid

from utils.data import produce_NA
from utils.eval import is_dag
from utils.graph import simulate_random_dag
import os
from utils.io import load_pickle, write_pickle
from sklearn.preprocessing import scale 


class SyntheticDataset:
    """Generate synthetic data.

    Key instance variables:
        X (numpy.ndarray): [n, d] data matrix.
        B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, root, n, d, config_code, graph_type, degree, noise_type,
                 miss_type='mcar', miss_percent=0.1, sem_type='linear',
                 equal_variances=True, mnar_type="logistic", p_obs=0.3, mnar_quantile_q=0.3):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            graph_type ('ER' or 'SF'): Type of graph.
            degree (int): Degree of graph.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            B_scale (float): Scaling factor for range of B.
            miss_percent (float): Percentage of missing data.
        """
        self.n = n
        self.d = d
        self.graph_type = graph_type
        self.degree = degree
        self.noise_type = noise_type
        self.miss_percent = miss_percent
        self.miss_type = miss_type
        self.sem_type = sem_type
        self.mnar_type = mnar_type
        self.p_obs = p_obs
        self.equal_variances = equal_variances
        self.mnar_quantile_q = mnar_quantile_q
        self.B_ranges = ((-2.0, -0.5), (0.5, 2.0))
        self.data_path = f'./{root}/{config_code}.pickle'

        self._setup()
        self._logger.debug("Finished setting up dataset class.")

    def _setup(self):
        """Generate B_bin, B and X."""
        if os.path.isfile(self.data_path):
            print('Loading data ...')
            self.B_bin, self.B, self.X_true, self.Omega, self.X, self.mask = load_pickle(self.data_path)
        else:
            print('Generating and Saving data ...')
            self.B_bin = simulate_random_dag(self.d,
                                             self.degree,
                                             self.graph_type)

            self.B = SyntheticDataset.simulate_weight(self.B_bin, self.B_ranges)
            if self.sem_type == 'linear':
                self.X_true, self.Omega = SyntheticDataset.simulate_linear_sem(
                    self.B, self.n, self.noise_type, self.equal_variances)
            else:
                self.X_true, self.Omega = SyntheticDataset.simulate_nonlinear_sem(
                    self.B_bin, self.n, self.sem_type, self.equal_variances)
            assert is_dag(self.B)

            if self.miss_percent > 0.0: 
                self.X, self.mask = produce_NA(self.X_true.copy(), p_miss=self.miss_percent, mecha=self.miss_type,
                                            opt=self.mnar_type, p_obs=self.p_obs, q=self.mnar_quantile_q)
            else:
                self.X, self.mask = None, None
            
            package = (self.B_bin, self.B, self.X_true, self.Omega, self.X, self.mask)
            write_pickle(package, self.data_path)

    # simulate_er_dag, simulate_sf_dag, simulate_random_dag removed
    # Use utils.graph.simulate_random_dag instead

    @staticmethod
    def simulate_weight(B_bin, B_ranges):
        """Simulate the weights of B_bin.

        Args:
            B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
            B_ranges (tuple): Disjoint weight ranges.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] weighted adjacency matrix of DAG.
        """
        B = np.zeros(B_bin.shape)
        S = np.random.randint(len(B_ranges), size=B.shape)  # Which range
        for i, (low, high) in enumerate(B_ranges):
            U = np.random.uniform(low=low, high=high, size=B.shape)
            B += B_bin * (S == i) * U
        return B

    @staticmethod
    def simulate_linear_sem(B, n, noise_type, equal_variances):
        """Simulate samples from linear SEM with specified type of noise.

        Args:
            B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
            n (int): Number of samples.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            equal_variances: if the variance is equal.

        Returns:
            numpy.ndarray: [n, d] data matrix.
        """

        def _simulate_single_equation(X, B_i, equal_variances):
            """Simulate samples from linear SEM for the i-th node.

            Args:
                X (numpy.ndarray): [n, number of parents] data matrix.
                B_i (numpy.ndarray): [d,] weighted vector for the i-th node.

            Returns:
                numpy.ndarray: [n,] data matrix.
            """
            scale = np.random.uniform(
                low=1.0, high=2.0) if not equal_variances else 1.0
            if noise_type == 'gaussian':
                # Gaussian noise
                N_i = np.random.normal(scale=scale, size=n)
            elif noise_type == 'exponential':
                # Exponential noise
                N_i = np.random.exponential(scale=scale, size=n)
            elif noise_type == 'gumbel':
                # Gumbel noise
                N_i = np.random.gumbel(scale=scale, size=n)
            elif noise_type == 'laplace':
                # Laplace noise
                N_i = np.random.laplace(scale=scale, size=n)
            elif noise_type == 'uniform':
                # Uniform noise
                N_i = np.random.uniform(low=-scale, high=scale, size=n)
            else:
                raise ValueError("Unknown noise type.")
            return X @ B_i + N_i, scale**2

        d = B.shape[0]
        X = np.zeros([n, d])
        Omega = np.zeros((d, d))    # Noise variance
        G = nx.DiGraph(B)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for i in ordered_vertices:
            parents = list(G.predecessors(i))
            X[:, i], Omega[i, i] = _simulate_single_equation(
                X[:, parents], B[parents, i], equal_variances)

        return X, Omega.astype(np.float32)

    @staticmethod
    def simulate_nonlinear_sem(B, n, sem_type, equal_variances=True, noise_scale=None):
        """Simulate samples from nonlinear SEM.
        Args:
            B (np.ndarray): [d, d] binary adj matrix of DAG
            n (int): num of samples
            sem_type (str): mlp, mim, gp, gp-add
            noise_scale (np.ndarray): scale parameter of additive noise, default all ones
        Returns:
            X (np.ndarray): [n, d] sample matrix
        """

        def _simulate_single_equation(X, scale, equal_variances):
            """X: [n, num of parents], x: [n]"""

            scale = np.random.uniform(
                low=1.0, high=2.0) if not equal_variances else 1.0
            z = np.random.normal(scale=scale, size=n)
            pa_size = X.shape[1]
            if pa_size == 0:
                return z
            if sem_type == 'mlp':
                hidden = 100
                W1 = np.random.uniform(
                    low=0.5, high=2.0, size=[pa_size, hidden])
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                W2[np.random.rand(hidden) < 0.5] *= -1
                x = sigmoid(X @ W1) @ W2 + z
            elif sem_type == 'mim':
                w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w1[np.random.rand(pa_size) < 0.5] *= -1
                w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w2[np.random.rand(pa_size) < 0.5] *= -1
                w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w3[np.random.rand(pa_size) < 0.5] *= -1
                x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
            elif sem_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = gp.sample_y(X, random_state=None).flatten() + z
            elif sem_type == 'gpadd':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                         for i in range(X.shape[1])]) + z
            else:
                raise ValueError('unknown sem type')
            return x

        d = B.shape[0]
        scale_vec = np.ones(d) if noise_scale is None else noise_scale
        Omega = np.diag(scale_vec**2)
        X = np.zeros([n, d])
        G = ig.Graph.Adjacency(B.tolist())
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            X[:, j] = _simulate_single_equation(
                X[:, parents], scale_vec[j], equal_variances)
        return X.astype(np.float32), Omega.astype(np.float32)
