import torch
import torch.nn as nn
import numpy as np

class WassersteinDist(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, X, Y):
        mu_X = torch.mean(X)
        mu_Y = torch.mean(Y)

        std_X = torch.std(X)
        std_Y = torch.std(Y)
        # print(f"mu_X={mu_X}\tstd_X={std_X}\nmu_Y={mu_Y}\tstd_Y={std_Y}")

        # Compute the squared difference between the means
        mean_diff_squared = (mu_X - mu_Y) ** 2
        
        # Compute the term for the variances
        variance_term = std_X ** 2 + std_Y ** 2 - 2 * torch.sqrt(std_X ** 2 * std_Y ** 2)
        
        # Compute the Wasserstein-1 distance
        loss = torch.sqrt(mean_diff_squared + variance_term)

        if self.reduction == "none":
            return -loss
        if self.reduction == "mean":
            return -loss.mean()
        return -loss.sum()

# X = torch.normal(mean=torch.tensor(2.0), std=torch.tensor(1.0), size=(100,))
# Y = torch.normal(mean=torch.tensor(5.0), std=torch.tensor(2.0), size=(100,))
# print(f"X = {X}")
# print(f"Y = {Y}")

# wsd = WassersteinDist()
# loss = wsd(X, Y)
# print(f"Wasserstein Distance = {loss}")