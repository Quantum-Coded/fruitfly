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

    def encode_single_letter(
        self,
        letter: Optional[str],
        position: int = 0,
        is_carrying: bool = False
    ) -> torch.Tensor:
        """
        Sensory drive when fly grasps or carries a specific letter.
        Injected into the specific ORN class (smell) for that letter + mushroom body memory.
        """
        rates = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        if letter:
            c_idx = ord(letter.upper()) - ord('A')
            if 0 <= c_idx < 26:
                drive_hz = 180.0 if not is_carrying else 145.0
                rates += self.letter_masks[c_idx].float() * drive_hz

        # Associative memory in Mushroom Body Kenyon Cells
        mb_sparse = (torch.rand(self.n, device=self.device) < 0.06) & self.is_mb
        rates += mb_sparse.float() * 65.0

        # If carrying while moving, descending motor neurons fire
        if is_carrying and self.is_descending is not None:
            rates += self.is_descending.float() * 80.0

        # Spontaneous biological baseline noise (~5 Hz)
        noise = torch.rand(self.n, device=self.device) < 0.03
        rates += noise.float() * 20.0

        return rates * EXT_DRIVE_SCALE

    def encode_tile_placement(
        self,
        letter: Optional[str],
        position: int = 0
    ) -> torch.Tensor:
        """
        Sensory confirmation burst when tile touches down on the board.
        Simultaneous ORN letter smell + visual position retinotopic confirmation + motor burst.
        """
        rates = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        # Visual retinotopic sector confirmation
        if 0 <= position < 5:
            rates += self.visual_sector_masks[position].float() * 190.0

        # Letter smell confirmation
        if letter:
            c_idx = ord(letter.upper()) - ord('A')
            if 0 <= c_idx < 26:
                rates += self.letter_masks[c_idx].float() * 170.0

        # Descending motor termination burst
        rates += self.is_descending.float() * 130.0

        # Mushroom body activity
        mb_sparse = (torch.rand(self.n, device=self.device) < 0.08) & self.is_mb
        rates += mb_sparse.float() * 80.0

        return rates * EXT_DRIVE_SCALE

    def encode_motor_walk(self, leg_phase: float = 0.0) -> torch.Tensor:
        """
        Descending motor neuron drive coordinated with walking leg gait.
        """
        rates = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        gait_mod = 0.8 + 0.5 * abs(np.sin(leg_phase))
        rates += self.is_descending.float() * (90.0 * gait_mod)

        # Baseline central brain spontaneous activity
        noise = torch.rand(self.n, device=self.device) < 0.035
        rates += noise.float() * 22.0

        return rates * EXT_DRIVE_SCALE

    def encode_revealing_colors(self, feedback: Optional[List[int]]) -> torch.Tensor:
        """
        Optic lobe visual neuron activation across 5 positions according to tile colors.
        """
        rates = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        if feedback:
            for pos, fb in enumerate(feedback[:5]):
                sec_mask = self.visual_sector_masks[pos].float()
                if fb == 2:    # Green (Correct)
                    rates += sec_mask * 220.0
                elif fb == 1:  # Yellow (Present)
                    rates += sec_mask * 135.0
                elif fb == 0:  # Gray (Absent)
                    rates += sec_mask * 35.0

        mb_sparse = (torch.rand(self.n, device=self.device) < 0.05) & self.is_mb
        rates += mb_sparse.float() * 70.0

        return rates * EXT_DRIVE_SCALE

    def encode_dopamine_reward(self, reward: float) -> torch.Tensor:
        """Dedicated dopamine pulse drive."""
        rates = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        if reward > 0:
            rates += self.is_dopamine.float() * (reward * 300.0)
        return rates * EXT_DRIVE_SCALE

