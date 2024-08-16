import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.4
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x



class AttentionMIL(nn.Module):
    def __init__(self, in_features, num_classes=2, L=512, D=128, n_leision = 5, attn_mode="gated", dropout_node=0.0):
        super().__init__()
        self.L = L
        self.D = D
        self.K = 1

        self.attn_mode = attn_mode

        
        self.MLP = nn.Sequential(
            nn.Linear(in_features, self.L),
            nn.ReLU(),
        )

        if attn_mode == 'gated':
            self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
            )

            self.attention_U = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Sigmoid()
            )

            self.attention_weights = nn.Linear(self.D, self.K)
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Tanh(),
                nn.Linear(self.D, self.K)
            )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_node) if dropout_node>0.0 else nn.Identity(),
            nn.Linear(self.L*self.K, num_classes)
        )

    def forward(self, x):
        H = self.MLP(x)  # NxL

        if self.attn_mode == 'gated':
            A_V = self.attention_V(H)  # NxD
            A_U = self.attention_U(H)  # NxD
            A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N
        else:
            A = self.attention(H)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        logits = self.classifier(M)

        return logits, A ,H


if __name__ == "__main__":
    milnet = AttentionMIL(512, attn_mode='linear', dropout_node=0.1)
    print(milnet)

    logits, A, _ = milnet(torch.randn(10, 512))
    print(logits.size(), A.size())