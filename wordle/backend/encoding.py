"""
encoding.py
Sensory encoding layer that maps Wordle observations into biological current injections
for the real Drosophila connectome.

Sensory modalities:
  1. Olfactory (Nose):
     - 26 ORN classes corresponding to letters A-Z (748 neurons).
     - Letter presentation injects current into the ORNs for the 5 letters of the word.
     - Drive rate ~120-180 Hz.
  2. Visual (Eyes):
     - Retinotopic visual projection neurons (406 neurons) mapped across 5 tile positions.
     - Color feedback modulation:
         * Green (Correct): High drive (180 Hz)
         * Yellow (Present): Medium drive (110 Hz)
         * Gray (Absent): Low/inhibitory baseline drive (30 Hz)
  3. Dopaminergic (Reward):
     - PAM/PPL1 dopamine neurons stimulated upon goal achievement or positive feedback.
"""

import torch
import numpy as np
from typing import List, Optional

EXT_DRIVE_SCALE = 0.012  # mV per Hz of injected sensory rate

class WordleSensoryEncoder:
    def __init__(self, circuit_data: dict, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.n = len(circuit_data['root_ids'])

        # Letter to ORN masks: list of 26 boolean tensors
        self.letter_masks = []
        is_orn_letter = circuit_data['is_orn_letter']
        for l_idx in range(26):
            mask = torch.tensor(is_orn_letter == l_idx, dtype=torch.bool, device=self.device)
            self.letter_masks.append(mask)

        # Visual sector masks: 5 positions (0 to 4)
        self.visual_sector_masks = []
        visual_sector = circuit_data['visual_sector']
        for sec in range(5):
            mask = torch.tensor(visual_sector == sec, dtype=torch.bool, device=self.device)
            self.visual_sector_masks.append(mask)

        # Mushroom body mask
        self.is_mb = torch.tensor(circuit_data['is_mb'], dtype=torch.bool, device=self.device)

        # Dopamine neuron mask
        self.is_dopamine = torch.tensor(circuit_data['is_dopamine'], dtype=torch.bool, device=self.device)

        # Descending neuron mask
        self.is_descending = torch.tensor(circuit_data['is_descending'], dtype=torch.bool, device=self.device)

    def encode_guess_and_feedback(
        self,
        guess: Optional[str] = None,
        feedback: Optional[List[int]] = None,
        dopamine_pulse: float = 0.0
    ) -> torch.Tensor:
        """
        Builds external drive tensor (size n) in mV for one simulation step.
        """
        rates = torch.zeros(self.n, dtype=torch.float32, device=self.device)

        # 1. Olfactory injection for letters in guess
        if guess:
            guess = guess.upper()
            for pos, char in enumerate(guess[:5]):
                c_idx = ord(char) - ord('A')
                if 0 <= c_idx < 26:
                    # Letter drive (base 140 Hz)
                    rates += self.letter_masks[c_idx].float() * 140.0

        # 2. Visual injection for tile positions & feedback colors
        if feedback:
            for pos, fb in enumerate(feedback[:5]):
                sec_mask = self.visual_sector_masks[pos].float()
                if fb == 2:    # Green (Correct)
                    rates += sec_mask * 180.0
                elif fb == 1:  # Yellow (Present)
                    rates += sec_mask * 110.0
                elif fb == 0:  # Gray (Absent)
                    rates += sec_mask * 25.0

        # 3. Dopamine reward injection
        if dopamine_pulse > 0.0:
            rates += self.is_dopamine.float() * (dopamine_pulse * 250.0)

        # Convert Hz to mV input drive
        return rates * EXT_DRIVE_SCALE

    def encode_dopamine_reward(self, reward: float) -> torch.Tensor:
        """Dedicated dopamine pulse drive."""
        rates = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        if reward > 0:
            rates += self.is_dopamine.float() * (reward * 300.0)
        return rates * EXT_DRIVE_SCALE
