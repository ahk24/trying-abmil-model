# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim, M=500, L=128, attention_branches=1):
        """
        Args:
            input_dim (int): Dimensionality of the input embeddings (e.g., 1024).
            M (int): Internal feature dimension.
            L (int): Hidden dimension for the attention mechanism.
            attention_branches (int): Number of attention branches.
        """
        super(Attention, self).__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = attention_branches

        # Replace convolutional layers with a simple linear projection.
        self.embedding_proj = nn.Sequential(
            nn.Linear(input_dim, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # projects to attention hidden space
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES)  # produces attention scores
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()  # output probability
        )

    def forward(self, x):
        # x is expected to have shape (1, num_instances, input_dim)
        x = x.squeeze(0)  # now shape: (num_instances, input_dim)
        H = self.embedding_proj(x)  # shape: (num_instances, M)

        A = self.attention(H)       # shape: (num_instances, ATTENTION_BRANCHES)
        A = A.transpose(1, 0)       # shape: (ATTENTION_BRANCHES, num_instances)
        A = F.softmax(A, dim=1)     # softmax over instances

        Z = torch.mm(A, H)          # aggregates to (ATTENTION_BRANCHES, M)
        Y_prob = self.classifier(Z) # (ATTENTION_BRANCHES, 1)
        Y_hat = (Y_prob >= 0.5).float()
        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        # Clamp the predictions for numerical stability.
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.-1e-5)
        neg_log_likelihood = - (Y * torch.log(Y_prob) + (1.-Y) * torch.log(1.-Y_prob))
        return neg_log_likelihood, A


class GatedAttention(nn.Module):
    def __init__(self, input_dim, M=500, L=128, attention_branches=1):
        super(GatedAttention, self).__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = attention_branches

        self.embedding_proj = nn.Sequential(
            nn.Linear(input_dim, self.M),
            nn.ReLU()
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (1, num_instances, input_dim)
        x = x.squeeze(0)  # shape: (num_instances, input_dim)
        H = self.embedding_proj(x)  # shape: (num_instances, M)
        A_V = self.attention_V(H)     # shape: (num_instances, L)
        A_U = self.attention_U(H)     # shape: (num_instances, L)
        A = self.attention_w(A_V * A_U)  # shape: (num_instances, ATTENTION_BRANCHES)
        A = A.transpose(1, 0)         # shape: (ATTENTION_BRANCHES, num_instances)
        A = F.softmax(A, dim=1)
        Z = torch.mm(A, H)            # shape: (ATTENTION_BRANCHES, M)
        Y_prob = self.classifier(Z)   # shape: (ATTENTION_BRANCHES, 1)
        Y_hat = (Y_prob >= 0.5).float()
        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.-1e-5)
        neg_log_likelihood = - (Y * torch.log(Y_prob) + (1.-Y) * torch.log(1.-Y_prob))
        return neg_log_likelihood, A
