import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional


def update_center_greedy(graphs, p, q=0, nV=None):
    """
    Computes DAG aggregation for the given graphs.

    Parameters
    ----------
    graphs : np.ndarray or list of np.ndarray
        - If np.ndarray: shape (N, N, K) adjacency matrices.
        - If list: list of edge lists (each M_i x 2).
    p : float
    q : float, optional
    nV : int, optional
        Number of vertices (only needed if input is a list of edge lists).

    Returns
    -------
    center : np.ndarray
        Aggregated DAG adjacency matrix (nV x nV).
    cost : int
        Currently unsupported, always -1.
    edges : np.ndarray
        Edge list of the aggregated DAG.
    """

    if isinstance(graphs, list) and nV is not None:
        # Case: list of edge lists
        edge_counts = np.zeros((nV, nV), dtype=int)
        all_edges = []
        M = len(graphs)

        for edge_list in graphs:
            for u, v in edge_list:
                if edge_counts[u-1, v-1] == 0:   # MATLAB is 1-based
                    all_edges.append((u-1, v-1))
                edge_counts[u-1, v-1] += 1

        all_edges = np.array(all_edges)
        regrets = np.zeros(all_edges.shape[0])

        for idx, (i, j) in enumerate(all_edges):
            regrets[idx] = (
                p * (2*edge_counts[i, j] + 2*edge_counts[j, i] - M)
                - edge_counts[j, i]
                + q * (M - edge_counts[i, j] - edge_counts[j, i])
            )

        mask = regrets > 0
        candidates = all_edges[mask]
        regrets = regrets[mask]
        order = np.argsort(-regrets)  # descending
        candidates = candidates[order]

    else:
        # Case: adjacency matrices
        graphs = np.array(graphs)
        graphs_tr = np.transpose(graphs, (1, 0, 2))
        ij = np.sum(graphs, axis=2)
        ji = np.sum(graphs_tr, axis=2)
        M = graphs.shape[2]

        c = p * (2*ij + 2*ji - M) - ji + q * (M - ij - ji)
        i, j = np.where(c > 0)
        candidates = np.vstack((i, j)).T
        values = c[c > 0]
        order = np.argsort(-values)
        candidates = candidates[order]
        nV = ij.shape[0]

    center = np.zeros((nV, nV), dtype=int)
    closed_center = np.zeros((nV, nV), dtype=int)

    # Iteratively add edges while maintaining acyclicity
    for i, j in candidates:
        if closed_center[j, i] == 0 and center[i, j] == 0:
            center[i, j] = 1
            if closed_center[i, j] == 0:
                closed_center[i, j] = 1
                in_nodes = np.where(closed_center[:, i])[0]
                out_nodes = np.where(closed_center[j, :])[0]

                closed_center[i, out_nodes] = 1
                closed_center[in_nodes, j] = 1
                for ii in in_nodes:
                    closed_center[ii, out_nodes] = 1

    cost = -1

    G = nx.from_numpy_array(center, create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(G):
        print("Non-cyclic graph!")
        raise ValueError("Graph is not a DAG")

    edges = np.argwhere(center == 1)
    return center, cost, edges


def dag_dist(graph1, graph2, p, q=0, edges1=None, edges2=None):
    """
    Distance between two directed acyclic graphs (DAGs).

    Parameters
    ----------
    graph1, graph2 : np.ndarray
        Adjacency matrices (nV x nV).
    p, q : float
        Parameters controlling distance metric.
    edges1, edges2 : np.ndarray, optional
        Edge lists (M x 2). If provided, distance is computed using edge lists
        (more efficient for sparse graphs).

    Returns
    -------
    dist : float
        Distance between the DAGs.
    """
    if edges1 is None or edges2 is None:
        # Case: adjacency matrices
        temp = graph1 * graph2.T
        D = np.sum(temp)  # discordant pairs

        deltaE = np.sum(np.abs(graph1 - graph2))

        temp2 = graph1 * graph2
        C = np.sum(temp2)  # concordant pairs

        n = graph1.shape[0]
        dist = p * (deltaE - 2*D) + D + q * (n*(n-1)/2 - (deltaE - D + C))
    else:
        # Case: edge lists
        n1 = edges1.shape[0]
        n2 = edges2.shape[0]

        if n1 < n2:
            edges = edges1
            graph = graph2
        else:
            edges = edges2
            graph = graph1

        case1, case2 = 0, 0
        for u, v in edges:
            if graph[u, v]:
                case1 += 1  # concordant
            elif graph[v, u]:
                case2 += 1  # discordant

        case3 = n1 - (case1 + case2) + n2 - (case1 + case2)

        nv = graph1.shape[0]
        case4 = (nv*nv - nv)//2 - case3 - case2 - case1  # semi-ambiguous

        dist = case2 + p*case3 + q*case4

    return dist


def update_center_median_greedy(graphs, p, q=0):
    """
    Computes DAG aggregation for given graphs by selecting the graph with the
    smallest distance to all others, then greedily adding new edges if it
    improves the center.

    Parameters
    ----------
    graphs : np.ndarray
        Shape (nV, nV, nGraphs), adjacency matrices.
    p, q : float
        Parameters controlling the aggregation.

    Returns
    -------
    center : np.ndarray
        Aggregated DAG adjacency matrix.
    cost : float
        Currently -1 (not computed).
    """
    graphs_tr = np.transpose(graphs, (1, 0, 2))
    ij = np.sum(graphs, axis=2)
    ji = np.sum(graphs_tr, axis=2)
    ij_and_ji_0 = np.sum((graphs == 0) & (graphs_tr == 0), axis=2)

    c = p * (ij + ji - ij_and_ji_0) - ji + q * ij_and_ji_0

    candidates = np.argwhere(c > 0)
    values = c[c > 0]
    order = np.argsort(-values)
    candidates = candidates[order]

    # Find the closest graph as initial center
    n = graphs.shape[2]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = dag_dist(graphs[:, :, i], graphs[:, :, j], p, q)
    D = D + D.T

    costs = np.sum(D, axis=1)
    c_idx = np.argmin(costs)
    cost = costs[c_idx]
    center = graphs[:, :, c_idx].copy()

    # Improve the center
    for i, j in candidates:
        if center[j, i] == 0 and center[i, j] == 0:
            center[i, j] = 1
            # Update the transitive closure
            out_nodes = np.where(center[j, :] == 1)[0]
            in_nodes = np.where(center[:, i] == 1)[0]
            center[i, out_nodes] = 1
            center[in_nodes, j] = 1

    cost = -1
    G = nx.from_numpy_array(center, create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(G):
        print("Non-cyclic center (closest greedy)")
        # raise ValueError("Graph is not a DAG")

    return center, cost

