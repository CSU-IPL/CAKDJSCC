import torch


def centered_kernel_alignment(X, Y, kernel='rbf', sigma=None):
    X = X.float().view(X.size(0), -1)
    Y = Y.float().view(X.size(0), -1)
    if X.size(0) != Y.size(0):
        raise ValueError("X and Y must have the same number of samples")
    n = X.size(0)

    def _rbf_gram(matrix, sigma):
        norm = torch.sum(matrix ** 2, dim=1, keepdim=True)
        dist_sq = norm + norm.t() - 2 * torch.mm(matrix, matrix.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)

        if sigma is None:
            mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
            flat_dist = dist_sq[mask]
            if flat_dist.numel() == 0:
                sigma_sq = 1.0
            else:
                sigma_sq = torch.median(flat_dist).clamp(min=1e-8)
            sigma = torch.sqrt(sigma_sq)

        return torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-8))

    if kernel == 'linear':
        K = torch.mm(X, X.t())
        L = torch.mm(Y, Y.t())
    elif kernel == 'rbf':
        K = _rbf_gram(X, sigma)
        L = _rbf_gram(Y, sigma)
    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")

    H = torch.eye(n, device=K.device) - 1/n
    K_centered = H @ K @ H
    L_centered = H @ L @ H

    similarity = torch.trace(K_centered @ L_centered)
    norm = torch.sqrt(torch.trace(K_centered @ K_centered) * torch.trace(L_centered @ L_centered))
    return similarity / (norm + 1e-8)

if __name__ == '__main__':
    rand_1 = (torch.randint(0, 255, [32, 10], dtype=torch.float32) / 255.).cuda()
    rand_2 = (torch.randint(0, 255, [32, 10], dtype=torch.float32) / 255.).cuda()
    cka = centered_kernel_alignment(rand_1, rand_2)
    print(cka)
