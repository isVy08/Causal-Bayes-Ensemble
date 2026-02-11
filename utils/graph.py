import abc
import numpy as np
import igraph as ig
import networkx as nx
from typing_extensions import Optional

import torch
import torch.distributions as td

def simulate_er_dag(d, degree):
    """Simulate ER DAG using NetworkX package.

    Args:
        d (int): Number of nodes.
        degree (int): Degree of graph.

    Returns:
        numpy.ndarray: [d, d] binary adjacency matrix of DAG.
    """
    def _get_acyclic_graph(B_und):
        return np.tril(B_und, k=-1)

    def _graph_to_adjmat(G):
        # return nx.to_numpy_matrix(G)
        return nx.to_numpy_array(G)

    p = float(degree) / (d - 1)
    # Probability for edge creation
    G_und = nx.generators.erdos_renyi_graph(n=d, p=p)
    B_und_bin = _graph_to_adjmat(G_und)    # Undirected
    B_bin = _get_acyclic_graph(B_und_bin)
    return B_bin


def simulate_sf_dag(d, degree):
    """Simulate SF DAG using igraph package.

    Args:
        d (int): Number of nodes.
        degree (int): Degree of graph.

    Returns:
        numpy.ndarray: [d, d] binary adjacency matrix of DAG.
    """
    m = int(round(degree / 2))
    # igraph does not allow passing RandomState object
    G = ig.Graph.Barabasi(n=d, m=m, directed=True)
    B_bin = np.array(G.get_adjacency().data)
    return B_bin

  
def simulate_random_dag(d, degree, graph_type):
    """Simulate random DAG.

    Args:
        d (int): Number of nodes.
        degree (int): Degree of graph.
        graph_type ('ER' or 'SF'): Type of graph.

    Returns:
        numpy.ndarray: [d, d] binary adjacency matrix of DAG.
    """
    def _random_permutation(B_bin):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(B_bin.shape[0]))
        return P.T @ B_bin @ P

    if graph_type == 'ER':
        B_bin = simulate_er_dag(d, degree)
    elif graph_type == 'SF':
        B_bin = simulate_sf_dag(d, degree)
    else:
        raise ValueError("Unknown graph type.")
    return _random_permutation(B_bin)

def fill_triangular(vec: torch.Tensor, upper: bool = False):
    """
    Args:
        vec: A tensor of shape (..., n(n-1)/2)
        upper: whether to fill the upper or lower triangle
    Returns:
        An array of shape (..., n, n), where the strictly upper (lower) triangle is filled from vec
        with zeros elsewhere
    """
    num_nodes = num_lower_tri_elements_to_n(vec.shape[-1])
    idxs = torch.triu_indices(num_nodes, num_nodes, offset=1, device=vec.device)
    output = torch.zeros(vec.shape[:-1] + (num_nodes, num_nodes), device=vec.device)
    output[..., idxs[0, :], idxs[1, :]] = vec
    return output if upper else output.transpose(-1, -2)

def num_lower_tri_elements_to_n(x: int):
    """
    Calculate the size of the matrix from the number of strictly lower triangular elements.

    We have x = n(n - 1) / 2 for some n
    n² - n - 2x = 0
    so n = (1 + √(1 + 8x)) / 2
    """
    val = int(np.sqrt(1 + 8 * x) + 1) // 2
    if val * (val - 1) != 2 * x:
        raise ValueError("Invalid number of lower triangular elements")
    return val

class AdjacencyDistribution(td.Distribution, abc.ABC):
    """
    Probability distributions over binary adjacency matrices for graphs.

    NOTE: Because we want to differentiate through samples (which are binary matrices),
    we add a `relaxed_sample` method. This method is still expected to return valid samples (binary matrices).

    Since the statistics of the relaxed distribution can be difficult to calculate,
    we choose to implement `relaxed_sample` as the relaxed distribution but report other statistics for
    the underlying distribution, as well as providing the usual `sample` method for it.

    Clearly this approximation could be invalid when the relaxed distribution is very different from
    the underlying one (at high temperatures).
    """

    support = td.constraints.independent(td.constraints.boolean, 1)

    def __init__(self, num_nodes: int, validate_args: Optional[bool] = None):
        assert num_nodes > 0, "Number of nodes in the graph must be greater than 0"
        self.num_nodes = num_nodes
        event_shape = torch.Size((num_nodes, num_nodes))

        super().__init__(event_shape=event_shape, validate_args=validate_args)

    @abc.abstractmethod
    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0):
        """
        Sample a binary adjacency matrix from the relaxed distribution (see NOTE in the class docstring).

        Args:
            sample_shape: the shape of the samples to return
            temperature: The temperature of the relaxed distribution
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size()):
        """
        Sample a binary adjacency matrix from the underlying distribution.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def entropy(self):
        """
        Return the entropy of the underlying distribution.

        Returns:
            A tensor of shape batch_shape, with the entropy of the distribution
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mean(self):
        """
        Return the mean of the underlying distribution.

        This will be a matrix with all entries in the interval [0, 1].

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mode(self):
        """
        Return the mode of the underlying distribution.

        This will be an adjacency matrix.

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, value: torch.Tensor):
        """
        Get the log probability of each tensor from the sample space

        Args:
            value: a binary matrix of shape value_shape + batch_shape + (n, n)
        Returns:
            A tensor of shape value_shape + batch_shape, with the log probabilities of each tensor in the batch.
        """
        raise NotImplementedError
