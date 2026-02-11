import torch
def free_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def sinkhorn(log_alpha, n_iters=20):
    """Approximate a permutation matrix using Sinkhorn normalization."""
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)  # Row normalization
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)  # Column normalization
    return torch.exp(log_alpha)

def permute_rows_soft(x, log_alpha=None, n_iters=20, tau=0.1):
    """
    x: Tensor of shape (N, D, D)
    Returns: Tensor of same shape with softly permuted rows,
            such that the diagonal is biased to be the max per row.
    """
    # We want to permute rows to maximize the diagonal entries
    # Define a 'score' matrix where higher values on diagonal are better
    # Use log_alpha as initial "scores" for assignment
    if log_alpha is None:
        log_alpha = x
    log_alpha = log_alpha / tau  # Temperature parameter
    P_soft = sinkhorn(log_alpha, n_iters=n_iters)  # (N, D, D), soft permutation matrix
    return torch.bmm(P_soft, x)
