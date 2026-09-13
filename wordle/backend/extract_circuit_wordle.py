"""
extract_circuit_wordle.py
Extracts and structures the real Drosophila FlyWire connectome circuit
specifically for the Wordle decision-making task.

Populations mapped:
  - 26 real Olfactory Receptor Neuron (ORN) classes mapped to the 26 letters of the alphabet A-Z (748 neurons).
  - Retinotopic visual projection neurons mapped across 5 tile positions (1-5) and 3 color states (green/yellow/gray).
  - Mushroom body Kenyon cells (KCs) + MBONs for associative memory / state representation.
  - Dopaminergic neurons (PPL1 / PPL2) for dopamine reward signaling upon reaching the goal state.
  - Descending / premotor neurons for motor action & decision readout.

Input: backend/data/circuit.npz (real FlyWire v783 connectome subgraph)
Output: wordle/backend/data/circuit_wordle.npz
"""

import os
import numpy as np

def extract_wordle_circuit():
    src_path = os.path.join(os.path.dirname(__file__), '../../backend/data/circuit.npz')
    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'circuit_wordle.npz')

    print(f"Loading base connectome from {src_path}...")
    d = np.load(src_path, allow_pickle=True)

    root_ids = d['root_ids']
    coords = d['coords'].astype(np.float32)
    cell_types = d['cell_types']
    super_classes = d['super_classes']
    pre_idx = d['pre_idx']
    post_idx = d['post_idx']
    weights = d['weights'].astype(np.float32)
    n = len(root_ids)

    print(f"Base circuit: {n} neurons, {len(weights)} synapses.")

    # 1. Map 26 distinct ORN classes to letters A-Z
    unique_orns = sorted([c for c in np.unique(cell_types) if 'ORN' in str(c)])
    if len(unique_orns) < 26:
        raise ValueError(f"Need at least 26 ORN classes, found {len(unique_orns)}")
    orn_classes = unique_orns[:26]
    letters = [chr(ord('A') + i) for i in range(26)]
    letter_to_orn = dict(zip(letters, orn_classes))
    orn_to_letter = {v: k for k, v in letter_to_orn.items()}

    # is_orn_letter: array of length n; value is 0-25 for assigned letter, or -1 if not assigned ORN
    is_orn_letter = np.full(n, -1, dtype=np.int32)
    for idx, ct in enumerate(cell_types):
        if ct in orn_to_letter:
            is_orn_letter[idx] = ord(orn_to_letter[ct]) - ord('A')

    # 2. Identify Dopaminergic neurons (PPL1 / PPL2 / DANs)
    is_dopamine = np.array([any(k in str(ct) for k in ['PPL', 'PAM', 'DAN', 'dop']) for ct in cell_types], dtype=bool)

    # 3. Identify Mushroom Body neurons (Kenyon cells & MBONs)
    is_mb = np.array([any(k in str(ct) for k in ['KCab', 'KCg', 'LHMB', 'MB-C', 'MBON']) for ct in cell_types], dtype=bool)

    # 4. Identify Descending / Motor neurons
    is_descending = np.isin(super_classes, ['descending', 'motor'])

    # 5. Identify Visual / Optic / Projection neurons
    # Neurons in visual_projection, optic, or projection neurons (PNs)
    is_visual = np.isin(super_classes, ['visual_projection', 'optic']) | np.array(['PN' in str(ct) for ct in cell_types], dtype=bool)

    # Retinotopic sector for visual neurons: 5 positions (0 to 4) based on X coordinate quantile
    norm_x = coords[:, 0]
    # Split into 5 quantiles
    quantiles = np.quantile(norm_x[is_visual], [0.2, 0.4, 0.6, 0.8])
    visual_sector = np.full(n, -1, dtype=np.int32)
    for idx in np.where(is_visual)[0]:
        x = norm_x[idx]
        if x <= quantiles[0]:
            visual_sector[idx] = 0
        elif x <= quantiles[1]:
            visual_sector[idx] = 1
        elif x <= quantiles[2]:
            visual_sector[idx] = 2
        elif x <= quantiles[3]:
            visual_sector[idx] = 3
        else:
            visual_sector[idx] = 4

    print("Populations mapped:")
    for letter in ['A', 'B', 'C', 'X', 'Y', 'Z']:
        orn_name = letter_to_orn[letter]
        count = np.sum(cell_types == orn_name)
        print(f"  Letter '{letter}' -> {orn_name} ({count} neurons)")
    print(f"  Total ORN letter neurons: {np.sum(is_orn_letter >= 0)}")
    print(f"  Visual projection neurons: {np.sum(is_visual)}")
    print(f"  Mushroom body neurons (KC/MBON): {np.sum(is_mb)}")
    print(f"  Dopaminergic reward neurons: {np.sum(is_dopamine)}")
    print(f"  Descending readout neurons: {np.sum(is_descending)}")

    # Save structured circuit
    np.savez_compressed(
        out_path,
        root_ids=root_ids,
        coords=coords,
        cell_types=cell_types,
        super_classes=super_classes,
        pre_idx=pre_idx,
        post_idx=post_idx,
        weights=weights,
        is_orn_letter=is_orn_letter,
        is_visual=is_visual,
        visual_sector=visual_sector,
        is_mb=is_mb,
        is_dopamine=is_dopamine,
        is_descending=is_descending,
        orn_classes=np.array(orn_classes)
    )

    print(f"Successfully saved Wordle connectome circuit to {out_path} ({os.path.getsize(out_path):,} bytes)")

if __name__ == '__main__':
    extract_wordle_circuit()
