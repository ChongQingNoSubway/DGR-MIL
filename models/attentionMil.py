import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class AttnMIL(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.L = 64
        self.D = 64
        self.K = 1

        self.feature_extractor = nn.Sequential(
          nn.Linear(in_features, 256),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(128, self.L),
          nn.ReLU(),
          nn.Dropout(0.5),
        )

        self.attn = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor(x) # NxL

        A = self.attn(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        #Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, A