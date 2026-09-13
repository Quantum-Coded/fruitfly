"""
brain_wordle.py
Leaky Integrate-and-Fire (LIF) network simulation for the real FlyWire Drosophila connectome
adapted for the Wordle decision and reward task.

Physiological constants match the published Drosophila connectome models:
  - tau_m = 10.0 ms (membrane time constant)
  - V_rest = -65.0 mV
  - V_th = -50.0 mV (threshold)
  - V_reset = -65.0 mV
  - refractory = 2.0 ms
  - dt = 0.2 ms (5 kHz internal integration rate)
"""

import os
import torch
import numpy as np
from typing import Tuple, List, Dict

DT_MS = 0.2
TAU_M_MS = 10.0
V_REST = -65.0
V_TH = -50.0
V_RESET = -65.0
REFRACTORY_MS = 2.0
W_SCALE = 0.045

class WordleBrain:
    def __init__(self, circuit_path: str = None, device: str = None):
        if circuit_path is None:
            circuit_path = os.path.join(os.path.dirname(__file__), 'data', 'circuit_wordle.npz')

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Wordle Connectome Brain on device: {self.device}...")

        d = np.load(circuit_path, allow_pickle=True)
        self.root_ids = d['root_ids']
        self.cell_types = d['cell_types']
        self.super_classes = d['super_classes']
        self.n = len(self.root_ids)

        # 3D Coordinates normalized to [-0.5, 0.5]
        coords = d['coords'].astype(np.float32)
        lo, hi = coords.min(0), coords.max(0)
        span = np.maximum(hi - lo, 1.0)
        self.raw_coords = coords
        self.norm_coords = (coords - lo) / span - 0.5
        self.coords_tensor = torch.tensor(self.norm_coords, device=self.device)

        # Populations masks
        self.is_orn_letter = torch.tensor(d['is_orn_letter'], device=self.device)
        self.is_visual = torch.tensor(d['is_visual'], device=self.device)
        self.visual_sector = torch.tensor(d['visual_sector'], device=self.device)
        self.is_mb = torch.tensor(d['is_mb'], device=self.device)
        self.is_dopamine = torch.tensor(d['is_dopamine'], device=self.device)
        self.is_descending = torch.tensor(d['is_descending'], device=self.device)

        # Indices
        self.motor_idx = torch.where(self.is_descending)[0]
        self.dopamine_idx = torch.where(self.is_dopamine)[0]
        self.mb_idx = torch.where(self.is_mb)[0]
        self.visual_idx = torch.where(self.is_visual)[0]
        self.orn_idx = torch.where(self.is_orn_letter >= 0)[0]

        # Synaptic weight matrix
        pre = torch.tensor(d['pre_idx'], dtype=torch.long)
        post = torch.tensor(d['post_idx'], dtype=torch.long)
        w = torch.tensor(d['weights'], dtype=torch.float32) * W_SCALE
        W = torch.zeros((self.n, self.n), dtype=torch.float32)
        W[post, pre] = w
        self.W = W.to(self.device)

        # Dynamic state variables
        self.reset_state()

        print(f"WordleBrain initialized: {self.n} neurons, {len(pre)} synapses. Descending neurons: {len(self.motor_idx)}.")

    def reset_state(self):
        """Resets membrane potentials and refractory states."""
        self.V = torch.full((self.n,), V_REST, device=self.device)
        self.refractory = torch.zeros(self.n, device=self.device)
        self.spikes = torch.zeros(self.n, device=self.device)
        self.motor_rate_ema = 0.0
        self.dopamine_level = 0.0

    def step(self, ext_drive: torch.Tensor, n_substeps: int = 25) -> Tuple[List[int], Dict]:
        """
        Advances the biological LIF network by n_substeps * DT_MS (default 25 * 0.2ms = 5.0ms).
        ext_drive: Tensor of shape (n,) containing injected current in mV.
        """
        dt = DT_MS
        decay_m = dt / TAU_M_MS
        all_spikes_count = torch.zeros(self.n, device=self.device)

        for _ in range(n_substeps):
            syn_input = self.W @ self.spikes
            not_refractory = self.refractory <= 0
            dV = decay_m * (V_REST - self.V) + syn_input + ext_drive
            self.V = torch.where(not_refractory, self.V + dV, self.V)
            fired = (self.V >= V_TH) & not_refractory
            self.V = torch.where(fired, torch.full_like(self.V, V_RESET), self.V)
            self.refractory = torch.where(fired, torch.full_like(self.refractory, REFRACTORY_MS),
                                           torch.clamp(self.refractory - dt, min=0))
            self.spikes = fired.float()
            all_spikes_count += self.spikes

        # Motor rate
        motor_rate = self.spikes[self.motor_idx].mean().item() if len(self.motor_idx) else 0.0
        self.motor_rate_ema = 0.7 * self.motor_rate_ema + 0.3 * motor_rate

        # Dopamine level
        da_spikes = self.spikes[self.dopamine_idx].mean().item() if len(self.dopamine_idx) else 0.0
        self.dopamine_level = 0.8 * self.dopamine_level + 0.2 * da_spikes

        # Active neuron indices for telemetry
        active_idx = torch.nonzero(all_spikes_count > 0, as_tuple=True)[0].tolist()

        telemetry = {
            "active_count": len(active_idx),
            "motor_rate": float(self.motor_rate_ema),
            "dopamine_level": float(self.dopamine_level),
            "spikes_orn": int(all_spikes_count[self.orn_idx].sum().item()),
            "spikes_visual": int(all_spikes_count[self.visual_idx].sum().item()),
            "spikes_mb": int(all_spikes_count[self.mb_idx].sum().item()),
            "spikes_dopamine": int(all_spikes_count[self.dopamine_idx].sum().item()),
            "spikes_motor": int(all_spikes_count[self.motor_idx].sum().item()),
        }

        return active_idx, telemetry

    def get_descending_rates(self) -> torch.Tensor:
        """Returns firing vector of descending neurons."""
        if len(self.motor_idx) == 0:
            return torch.zeros(16, device=self.device)
        return self.spikes[self.motor_idx]
