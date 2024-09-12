"""Weighted Kappa Loss with KL-divergence."""

import torch
import torch.nn as nn


class kl_WeightedKappaLoss(nn.Module):
    """Weighted Kappa Loss.

    This class implements the weighted kappa loss function.

    Args:
        num_classes (int): Number of classes.
        weight_power (int): Weighting type.
        epsilon (float): Epsilon value to avoid division by zero.
    """

    def __init__(self, device, num_classes=5, weight_power=0.5, epsilon=1e-6):
        super(kl_WeightedKappaLoss, self).__init__()

        self.weight_power = weight_power
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device

        if weight_power > 2 or weight_power < 0:
            raise ValueError("Unknown kappa weighting type.")
        self.weight_matrix = torch.zeros(
            num_classes, num_classes, device=self.device)
        for i in range(num_classes):
            for j in range(num_classes):
                self.weight_matrix[i, j] = (
                    abs((i - j)) ** weight_power) / (
                    (num_classes - 1) ** weight_power
                    )
        if weight_power == 0:
            self.weight_matrix = torch.ones(
                num_classes, num_classes, device=self.device) \
                    - torch.eye(num_classes, device=self.device)

    def forward(self, y_pred, y_true):
        """Forward pass.

        Args:
            y_pred (torch.Tensor): Predicted labels.
            y_true (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Loss value.
        """
        batch_size = y_pred.shape[0]

        conf_mat = torch.zeros(
            self.num_classes,
            self.num_classes,
            device=self.device,
            dtype=torch.float32)
        chance_matrix = torch.zeros(
            self.num_classes,
            self.num_classes,
            device=self.device,
            dtype=torch.float32)

        for batch_idx in range(batch_size):
            true_label = y_true[batch_idx]
            pred_probs = y_pred[batch_idx]
            conf_mat[true_label] += pred_probs

        row_sums = conf_mat.sum(dim=1)
        col_sums = conf_mat.sum(dim=0)
        total_sum = conf_mat.sum()

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                expected_value = (row_sums[i] * col_sums[j]) / total_sum
                chance_matrix[i, j] = expected_value

        numerator = torch.sum(self.weight_matrix * conf_mat)
        if numerator < 0:
            print("Conf =", conf_mat)
            print("Weight", self.weight_matrix)
            print("Chance", chance_matrix)
            print("Y_pred=", y_pred)
            print("Y_true=", y_true)
            raise ValueError(f"Numerator is {numerator}.")
        denominator = torch.sum(self.weight_matrix * chance_matrix)
        if denominator <= 0:
            raise ValueError(f"Denominator is {denominator}.")

        kappa = 1 - (numerator / denominator)
        if kappa > 1 or kappa < -1:
            raise ValueError("Kappa score is outside of bounds.")

        penalty = (1 / total_sum) * torch.sum(
            row_sums * (
                torch.log(row_sums + torch.tensor(1e-8)) - torch.log(
                    col_sums + torch.tensor(1e-8))))

        loss = - torch.log((kappa + 1)/2)
        return loss + penalty
