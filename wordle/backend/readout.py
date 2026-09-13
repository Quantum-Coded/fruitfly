"""
readout.py
Trained neural decision readout head for the FlyWordle agent.

Maps descending / motor activity and candidate features to scalar policy logits.
The 3,127-neuron real connectome is frozen; this small readout layer is trained
via REINFORCE policy gradient gated by the real dopamine reward signal.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class WordleReadout(nn.Module):
    def __init__(self, n_descending: int = 13, n_features: int = 26 + 5 + 1):
        super().__init__()
        # Features: descending firing rates (13) + word letter bag (26) + positional entropy (5) + heuristic score (1)
        in_dim = n_descending + n_features
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, brain_descending: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        """
        brain_descending: (n_descending,)
        candidate_features: (batch_candidates, n_features)
        Returns: (batch_candidates,) scalar logits
        """
        # Broadcast descending activity to match candidate batch size
        desc_expanded = brain_descending.unsqueeze(0).expand(candidate_features.size(0), -1)
        x = torch.cat([desc_expanded, candidate_features], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        scores = self.out(h).squeeze(-1)
        return scores

    @staticmethod
    def extract_word_features(word: str, device: str = 'cpu') -> torch.Tensor:
        """
        Extracts feature vector for a 5-letter candidate word:
          - 26-dim one-hot / bag of letters
          - 5-dim positional vowel/consonant indicators
          - 1-dim unique letter ratio
        Total: 32 dimensions
        """
        feat = torch.zeros(32, dtype=torch.float32, device=device)
        word = word.upper()

        # Letter bag
        for c in word:
            idx = ord(c) - ord('A')
            if 0 <= idx < 26:
                feat[idx] += 1.0

        # Positional vowel indicator
        vowels = {'A', 'E', 'I', 'O', 'U', 'Y'}
        for pos, c in enumerate(word[:5]):
            if c in vowels:
                feat[26 + pos] = 1.0

        # Unique letter ratio
        feat[31] = len(set(word)) / 5.0
        return feat

    def score_candidates(
        self,
        brain_descending: torch.Tensor,
        candidates: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scores a list of candidate words given descending rates.
        Returns: (probs, logits)
        """
        device = brain_descending.device
        feat_list = [self.extract_word_features(w, device=device) for w in candidates]
        candidate_features = torch.stack(feat_list, dim=0)

        logits = self.forward(brain_descending, candidate_features)
        probs = F.softmax(logits, dim=0)
        return probs, logits
