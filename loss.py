import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedLoss(nn.Module):
    """
    Referenced to
    - "Class-Balanced Loss Based on Effective Number of Samples"
    - https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
    """

    def __init__(self, samples_per_cls, no_of_classes, loss_type="focal", beta=0.9999, gamma=2.0):
        loss_types = ["focal", "sidmoid", "softmax"]
        assert loss_type.lower() in loss_types, f"loss_type must be one of {loss_types} not {loss_type}"

        if not isinstance(samples_per_cls, torch.Tensor):
            samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float32)

        super().__init__()

        self.effective_num = 1.0 - beta ** samples_per_cls
        self.weights = (1.0 - beta) / self.effective_num
        # TODO no_of_classes를 왜 곱하는지??
        self.weights = self.weights / torch.sum(self.weights) * no_of_classes
        self.weights.unsqueeze_(0)

        self.no_of_classes = no_of_classes
        self.loss_type = loss_type.lower()
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target):
        one_hot = F.one_hot(target, self.no_of_classes).float()

        weights = self.weights.repeat(one_hot.size(0), 1) * one_hot
        weights = weights.sum(1).unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = self.focal_loss(one_hot, input, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input, one_hot, weight=self.weights)
        elif self.loss_type == "softmax":
            pred = torch.softmax(input, dim=1)
            cb_loss = F.binary_cross_entropy(pred, one_hot, weight=weights)
        else:
            raise f"Unknown loss_type {self.loss_type}"

        return cb_loss

    @staticmethod
    def focal_loss(labels, logits, alpha, gamma):
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
        labels: A float tensor of size [batch, num_classes].
        logits: A float tensor of size [batch, num_classes].
        alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
        gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
        focal_loss: A float32 scalar representing normalized total loss.
        """
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss
