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
        # Ensure EVERY letter A-Z has at least 35-45 active olfactory neurons
        self.letter_masks = []
        is_orn_letter = circuit_data['is_orn_letter']
        orn_indices = np.where(is_orn_letter >= 0)[0]

        for l_idx in range(26):
            direct_mask = (is_orn_letter == l_idx)
            count = int(np.sum(direct_mask))
            if count < 35:
                # Augment with deterministic pseudo-random subset of other ORN cells
                rng = np.random.RandomState(100 + l_idx * 17)
                needed = 38 - count
                extra = rng.choice(orn_indices, size=min(needed, len(orn_indices)), replace=False)
                combined = np.copy(direct_mask)
                combined[extra] = True
                mask = torch.tensor(combined, dtype=torch.bool, device=self.device)
            else:
                mask = torch.tensor(direct_mask, dtype=torch.bool, device=self.device)
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
                    rates += self.letter_masks[c_idx].float() * 160.0

        # 2. Visual injection for tile positions & feedback colors
        if feedback:
            for pos, fb in enumerate(feedback[:5]):
                sec_mask = self.visual_sector_masks[pos].float()
                if fb == 2:    # Green (Correct)
                    rates += sec_mask * 220.0
                elif fb == 1:  # Yellow (Present)
                    rates += sec_mask * 140.0
                elif fb == 0:  # Gray (Absent)
                    rates += sec_mask * 35.0

        # 3. Dopamine reward injection
        if dopamine_pulse > 0.0:
            rates += self.is_dopamine.float() * (dopamine_pulse * 280.0)

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
                drive_hz = 210.0 if not is_carrying else 175.0
                rates += self.letter_masks[c_idx].float() * drive_hz

        # Associative memory in Mushroom Body Kenyon Cells (70-90 neurons)
        mb_sparse = (torch.rand(self.n, device=self.device) < 0.18) & self.is_mb
        rates += mb_sparse.float() * 95.0

        # If carrying while moving, descending motor neurons fire
        if is_carrying and self.is_descending is not None:
            rates += self.is_descending.float() * 110.0

        # Spontaneous biological baseline noise (~5 Hz)
        noise = torch.rand(self.n, device=self.device) < 0.035
        rates += noise.float() * 25.0

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
        # Visual retinotopic sector confirmation (81 neurons)
        if 0 <= position < 5:
            rates += self.visual_sector_masks[position].float() * 230.0

        # Letter smell confirmation (38 neurons)
        if letter:
            c_idx = ord(letter.upper()) - ord('A')
            if 0 <= c_idx < 26:
                rates += self.letter_masks[c_idx].float() * 190.0

        # Descending motor termination burst
        rates += self.is_descending.float() * 160.0

        # Mushroom body activity
        mb_sparse = (torch.rand(self.n, device=self.device) < 0.20) & self.is_mb
        rates += mb_sparse.float() * 110.0

        return rates * EXT_DRIVE_SCALE

    def encode_motor_walk(self, leg_phase: float = 0.0) -> torch.Tensor:
        """
        Descending motor neuron drive coordinated with walking leg gait.
        """
        rates = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        gait_mod = 0.8 + 0.6 * abs(np.sin(leg_phase))
        rates += self.is_descending.float() * (130.0 * gait_mod)

        # Baseline central brain spontaneous activity
        noise = torch.rand(self.n, device=self.device) < 0.04
        rates += noise.float() * 28.0

        return rates * EXT_DRIVE_SCALE

    def encode_single_tile_reveal(self, pos: int, fb_code: int) -> torch.Tensor:
        """
        Focused Optic Lobe sensory drive when revealing color feedback for tile at 'pos' (0..4).
        Green (2) -> 260 Hz (Optic lobe massive burst)
        Yellow (1) -> 170 Hz (Optic lobe present burst)
        Gray (0) -> 45 Hz (Optic lobe absent baseline)
        """
        rates = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        if 0 <= pos < 5:
            sec_mask = self.visual_sector_masks[pos].float()
            if fb_code == 2:    # Green (Correct)
                rates += sec_mask * 260.0
                # Mushroom body positive association
                mb_sparse = (torch.rand(self.n, device=self.device) < 0.16) & self.is_mb
                rates += mb_sparse.float() * 120.0
            elif fb_code == 1:  # Yellow (Present)
                rates += sec_mask * 170.0
                mb_sparse = (torch.rand(self.n, device=self.device) < 0.09) & self.is_mb
                rates += mb_sparse.float() * 85.0
            elif fb_code == 0:  # Gray (Absent)
                rates += sec_mask * 45.0

        # Background spontaneous cortical noise
        noise = torch.rand(self.n, device=self.device) < 0.03
        rates += noise.float() * 22.0

        return rates * EXT_DRIVE_SCALE

    def encode_revealing_colors(self, feedback: Optional[List[int]]) -> torch.Tensor:
        """
        Optic lobe visual neuron activation across all 5 positions.
        """
        rates = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        if feedback:
            for pos, fb in enumerate(feedback[:5]):
                sec_mask = self.visual_sector_masks[pos].float()
                if fb == 2:    # Green (Correct)
                    rates += sec_mask * 240.0
                elif fb == 1:  # Yellow (Present)
                    rates += sec_mask * 150.0
                elif fb == 0:  # Gray (Absent)
                    rates += sec_mask * 40.0

        mb_sparse = (torch.rand(self.n, device=self.device) < 0.08) & self.is_mb
        rates += mb_sparse.float() * 80.0

        return rates * EXT_DRIVE_SCALE

    def encode_dopamine_reward(self, reward: float) -> torch.Tensor:
        """Dedicated dopamine pulse drive."""
        rates = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        if reward > 0:
            rates += self.is_dopamine.float() * (reward * 320.0)
            # Mushroom body broad plasticity burst
            mb_sparse = (torch.rand(self.n, device=self.device) < 0.25) & self.is_mb
            rates += mb_sparse.float() * 140.0
        return rates * EXT_DRIVE_SCALE

