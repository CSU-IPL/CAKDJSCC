import torch
import torch.nn as nn
import torch.nn.functional as F


def HSIC(X, Y):
    n = X.size(0)
    assert Y.size(0) == n, "X and Y must have the same number of samples"

    XX = torch.sum(X ** 2, dim=1, keepdim=True)
    XY = torch.matmul(X, X.t())
    pairwise_dists_X = XX - 2 * XY + XX.t()

    YY = torch.sum(Y ** 2, dim=1, keepdim=True)
    YZ = torch.matmul(Y, Y.t())
    pairwise_dists_Y = YY - 2 * YZ + YY.t()

    median_sq = torch.median(pairwise_dists_X.flatten())
    sigma_x = torch.sqrt(median_sq / 2) if median_sq != 0 else torch.tensor(1.0)
    median_sq_Y = torch.median(pairwise_dists_Y.flatten())
    sigma_y = torch.sqrt(median_sq_Y / 2) if median_sq_Y != 0 else torch.tensor(1.0)

    sigma_x = sigma_x.to(X.device)
    sigma_y = sigma_y.to(Y.device)

    gamma_x = 1.0 / (2 * sigma_x ** 2)
    gamma_y = 1.0 / (2 * sigma_y ** 2)
    K = torch.exp(-gamma_x * pairwise_dists_X)
    L = torch.exp(-gamma_y * pairwise_dists_Y)
    K_row_means = K.mean(dim=1, keepdim=True)
    K_col_means = K.mean(dim=0, keepdim=True)
    K_mean = K.mean()
    Kc = K - K_row_means - K_col_means + K_mean
    L_row_means = L.mean(dim=1, keepdim=True)
    L_col_means = L.mean(dim=0, keepdim=True)
    L_mean = L.mean()
    Lc = L - L_row_means - L_col_means + L_mean

    hsic = torch.sum(Kc * Lc) / (n - 1) ** 2

    return hsic


class InfoNCE(nn.Module):
    def __init__(self, device, temp=0.2):
        super().__init__()
        self.temp = temp
        self.device = device

    def forward(self, S, T):
        logits = torch.mm(S, T.T) / self.temp

        labels = torch.arange(S.size(0)).to(S.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class IBLoss(nn.Module):
    def __init__(self, temp, beta, device):
        super().__init__()
        self.temp = temp
        self.beta = beta
        self.device = device

    def forward(self, S, T, S_, X):
        T = T.detach()
        # 教师特征中心化
        T = T - T.mean(dim=0, keepdim=True)

        # 特征归一化
        T = F.normalize(T, dim=1)
        S = F.normalize(S, dim=1)

        L_NCE = InfoNCE(self.device, self.temp)(S, T)
        L_HSIC = HSIC(S_, X)
        IB_Loss = L_NCE + self.beta * L_HSIC
        return IB_Loss


class CNCE(nn.Module):
    def __init__(self, device, temp=0.1):
        super().__init__()
        self.temp = temp
        self.device = device

    def forward(self, S, T, Y, g_nce, g_bo):
        q_s = g_nce(S)
        k_t = g_nce(T)
        q_y = g_nce(Y)

        q_bo_s = g_bo(S)
        q_bo_y = g_bo(Y)

        s_s_t = torch.mm(q_s, k_t.T)
        s_y_t = torch.mm(q_y, k_t.T)

        nce_s = F.cross_entropy(s_s_t, torch.arange(S.size(0), device=S.device))
        nce_y = F.cross_entropy(s_y_t, torch.arange(S.size(0), device=S.device))

        with torch.no_grad():
            s_bo_s = torch.mm(q_bo_s, k_t.T)
            s_bo_y = torch.mm(q_bo_y, k_t.T)

        boosted_s = (s_y_t.detach()+s_bo_s)/self.temp
        boosted_y = (s_s_t.detach()+s_bo_y)/self.temp

        loss_bo_s = F.cross_entropy(boosted_s, torch.arange(S.size(0), device=S.device))
        loss_bo_y = F.cross_entropy(boosted_y, torch.arange(S.size(0), device=S.device))

        return nce_s + nce_y + loss_bo_y + loss_bo_s


class CIBLoss(nn.Module):
    def __init__(self, temp, alpha, beta, device):
        super().__init__()
        self.temp = temp
        self.beta = beta
        self.alpha = alpha
        self.device = device

    def forward(self, S, T, S_, X, Y, g_nce, g_bo):
        L_NCE = InfoNCE(self.device, self.temp)(S, T)
        L_HSIC = HSIC(S_, X)
        L_CNCE = CNCE(self.device, self.temp)(S, T, Y, g_nce, g_bo)
        CIB_Loss = 4 * L_NCE + self.alpha * L_CNCE + self.beta * L_HSIC
        return CIB_Loss
