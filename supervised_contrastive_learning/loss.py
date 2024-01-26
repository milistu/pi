import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.10) -> None:
        super(SupervisedContrastiveLoss).__init__()
        self.temperature = temperature

    def forward(self, features, labels) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.unsqueeze(1)
        mask_positive = torch.eq(labels, labels.T).float()
        mask_positive.fill_diagonal_(0)

        exp_similarities = torch.exp(similarity_matrix)

        sum_exp_similarities = torch.sum(
            exp_similarities * (1 - mask_positive), dim=1, keepdim=True
        )

        loss = -torch.sum(
            torch.log(exp_similarities / sum_exp_similarities) * mask_positive, dim=1
        )

        num_positive = mask_positive.sum(dim=1)

        loss = loss / num_positive.clamp(min=1)

        return loss.mean()
