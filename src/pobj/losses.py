"""The two pretraining losses, and the combination (section 7.5)."""
import torch.nn as nn
import torch.nn.functional as F


def letter_loss(logits, targets):
    """Arm C. logits: [M, 20], targets: [M] -> scalar"""
    return F.cross_entropy(logits, targets)


def vector_loss(predicted, target):
    """Arm B. Both [M, dim] -> scalar.

    One minus cosine similarity, so only the direction of the embedding is matched. The
    target carries no gradient; it comes from the teacher.
    """
    return (1 - F.cosine_similarity(predicted, target.detach(), dim=-1)).mean()


def combined(letter, vector, lam):
    """Arm D."""
    return letter + lam * vector
