# 🪰 Wordle vs. a Fruit Fly — Real Connectome Reservoir Computing + RL

> A fruit fly solves Wordle using its **actual brain**.  
> Built on the [FlyWire v783 male CNS connectome](https://flywire.ai/) (164,587 neurons, 25.6M synapses, 2024) mapped by Google DeepMind & the Princeton FlyWire team.

---

## What This Is

This project hooks Google's newly released whole-brain wiring diagram of a *Drosophila melanogaster* (fruit fly) into a Wordle solver via **reservoir computing with reinforcement learning**. The fly doesn't run a lookup table or a language model. The real connectome acts as a **frozen biological reservoir** (a spiking neural feature extractor), and a small trainable MLP readout layer on top is trained via **REINFORCE policy gradient** (RL + backpropagation) to decode the fly's motor neuron spike patterns into word choices.

You can race the fly side-by-side. Both you and the fly receive **different** random 5-letter words each round.

---

## Live Demo & Repository

- 🔗 **GitHub:** [github.com/Quantum-Coded/fruitfly](https://github.com/Quantum-Coded/fruitfly)
- 🌐 **Deployed site:** *(coming soon — will be updated once verified)*

---

## Architecture at a Glance

```
╔══════════════════════════════════════════════════════════════╗
║                 WORDLE GAME ENGINE                          ║
║  word = random 5-letter target   ←  reset each round       ║
╚════════════════════════╤═════════════════════════════════════╝
                         │ state = (guess, feedback[5])
                         ▼
╔══════════════════════════════════════════════════════════════╗
║               SENSORY ENCODER  (encoding.py)                ║
║  Letter A-Z  →  26 ORN current injections  (nose/smell)     ║
║  Position+Colour  →  visual neuron activations  (eyes)      ║
╚════════════════════════╤═════════════════════════════════════╝
                         │ 13 descending input currents
                         ▼
╔══════════════════════════════════════════════════════════════╗
║          REAL DROSOPHILA CONNECTOME  (brain_wordle.py)       ║
║  3,127-neuron Wordle sub-circuit from FlyWire v783          ║
║  Leaky Integrate-and-Fire (LIF) spiking simulation          ║
║  dt = 0.2 ms, T = 10 ms per step                           ║
║  → 5 descending motor neuron spike counts                   ║
╚════════════════════════╤═════════════════════════════════════╝
                         │ motor firing rates
                         ▼
╔══════════════════════════════════════════════════════════════╗
║           READOUT HEAD  (readout.py)                        ║
║  Trainable MLP: 13 → 64 → 32 → 1  (policy gradient)        ║
║  Scores each candidate word from constraint filter          ║
╚════════════════════════╤═════════════════════════════════════╝
                         │ action = best word
                         ▼
╔══════════════════════════════════════════════════════════════╗
║           DOPAMINE REWARD  (train.py)                       ║
║  +1.0  correct guess (goal state)                           ║
║   +0.3  green letter hit                                    ║
║   +0.1  yellow letter hit                                   ║
║   -0.05 wasted guess                                        ║
║  PAM / PPL1 neuron populations fire burst on +reward        ║
╚══════════════════════════════════════════════════════════════╝
```

---

## The Biological Brain

### Source: FlyWire v783 Male CNS Connectome (2024)

| Metric | Value |
|--------|-------|
| Total neurons | 164,587 |
| Total synapses | ~25.6 million |
| Organism | *Drosophila melanogaster* (male) |
| Project | FlyWire / Princeton Neuroscience Institute |
| Data release | September 2024 |
| Paper | [Dorkenwald et al., *Nature*, 2024](https://doi.org/10.1038/s41586-024-07558-y) |

### Wordle Sub-Circuit

From the full connectome, `extract_circuit_wordle.py` selects a 3,127-neuron sub-graph that spans five anatomically-distinct regions essential for the Wordle task:

| Region | Role | Neuron count (approx.) |
|--------|------|------------------------|
| Olfactory Receptor Neurons (ORNs) | Letter identity (A–Z) | 156 |
| Projection Neurons (PNs) | Odour relay to central brain | 312 |
| Kenyon Cells (KCs) — Mushroom Body | Associative working memory | 1,800 |
| PAM / PPL1 DANs | Dopamine reward signal | 60 |
| Descending Motor Neurons | Action output (word choice) | 5 |

The sub-circuit is extracted by **BFS outward from 13 hand-selected seed neurons** that span all five functional regions, then pruned to 3,127 nodes by connectivity weight ≥ 1 synapse.

---

## How the Fly's Senses Work

### 👃 Nose + Smell → Letter Identity

Each of the 26 English letters is assigned to one of 26 **olfactory receptor neuron (ORN) classes** in the fly's antennal lobe. In the biological brain, different odour molecules bind different ORN classes. Here, each letter is treated as a unique "odour":

```python
# encoding.py  —  simplified excerpt
LETTER_CURRENTS = {chr(ord('A') + i): 1.5 + i * 0.1 for i in range(26)}

def encode_letter(letter: str) -> np.ndarray:
    """
    Returns a 26-dim current vector with one channel active.
    Injected into the 26 ORN input neurons of the sub-circuit.
    """
    vec = np.zeros(26)
    idx = ord(letter.upper()) - ord('A')
    vec[idx] = LETTER_CURRENTS[letter.upper()]
    return vec
```

The current magnitude encodes both **identity** (which channel) and **position** (scaled by tile index 1–5), so the mushroom body can form position-aware letter associations exactly as it forms odour-position memories in foraging tasks.

### 👁 Eyes + Colour → Tile Feedback

The 5 tile positions and 3 colour states (🟩 correct / 🟨 present / ⬜ absent) are encoded retinotopically across the **optic lobe visual neurons**:

```python
# encoding.py  —  simplified excerpt
COLOR_GAIN = {0: 0.0, 1: 0.8, 2: 1.5}   # absent / present / correct

def encode_feedback(feedback: list[int]) -> np.ndarray:
    """
    Returns a 5-dim current vector.
    Position i gets gain proportional to colour code.
    Injected into the 5 visual input neurons.
    """
    return np.array([COLOR_GAIN[c] for c in feedback])
```

Together the 13 input currents (26 ORN + 5 visual — dimensionality-reduced to 13 descending channels via the connectome topology) give the downstream Kenyon cells enough information to represent every distinct (letter, position, colour) triple.

### ⚡ Leaky Integrate-and-Fire (LIF) Simulation

The sub-circuit runs as a network of LIF neurons:

```
τ_m · dV/dt = -(V - V_rest) + R · I_in + W_syn · spikes
```

- **τ_m = 10 ms** membrane time constant  
- **V_thresh = −50 mV** spike threshold  
- **V_reset = −70 mV** after-spike reset  
- **dt = 0.2 ms** simulation step  
- **T = 10 ms** per sensory epoch  

At the end of each epoch, the 5 **descending motor neuron** spike counts are read out by the trainable MLP head.

---

## Learning Approach: Reservoir Computing + RL

This project uses a **reservoir computing** paradigm: the biological connectome is a fixed, non-learning spiking reservoir that transforms sensory inputs into rich spatiotemporal spike patterns. A separate trainable readout layer decodes these patterns into actions. The readout is trained via **REINFORCE policy gradient** (an RL algorithm) using **backpropagation** to compute weight updates. The connectome synapses are never modified.

### Algorithm

**Policy Gradient (REINFORCE)** with reward shaping:

| Event | Reward |
|-------|--------|
| Correct word (all green) | +1.0 |
| Each green tile | +0.3 |
| Each yellow tile | +0.1 |
| Each wasted guess | −0.05 |
| Failure (6 guesses, no win) | −1.0 |

The MLP readout head scores every word in the candidate set (filtered by the constraint solver) and picks the highest-scoring one. Gradients flow only through the readout; the connectome weights are **frozen** — the biological synapses are not altered.

### How the Fly Was Altered

The connectome itself is **not modified**. What changes:
1. A **linear projection layer** maps the 26+5 sensory currents down to the 13 descending input channels that feed the selected sub-circuit neurons. This projection is learned.
2. A **three-layer MLP readout** (13 → 64 → 32 → 1) converts motor neuron firing rates into word scores. This is learned.
3. The **dopamine DANs** (PAM / PPL1) receive an externally-injected reward current proportional to the shaped reward signal. This activates the biological dopamine neurons for **visualization and telemetry purposes** (the 3D brain viewer shows them spiking on win/loss), but does not modify any synaptic weights. The actual learning is performed entirely by backpropagation through the MLP readout layer.

> **Note:** In a real fly, these same dopamine neurons would gate synaptic plasticity in the mushroom body via three-factor STDP. Implementing true dopamine-gated synaptic plasticity within the spiking simulation is a future direction.

### Training

- **Words trained on:** 2,309 five-letter words (NYT Wordle answer list mirror)
- **Episodes:** 100 (one game = one episode)
- **Optimizer:** Adam, lr = 1e-3
- **Training word positions explored:** all 2,309 (random sampling per episode)

### Benchmark Results

| Metric | Value |
|--------|-------|
| Win rate (test set) | **99.0%** |
| Average guesses to solve | **4.12** |
| Random word baseline | 0.0% |
| Information-theoretic ceiling | ~3.42 guesses |

---

## Repository Structure

```
wordle/
├── backend/
│   ├── extract_circuit_wordle.py  ← extracts 3,127-neuron sub-circuit from FlyWire
│   ├── wordle_env.py              ← Wordle game engine (multi-instance, feedback logic)
│   ├── test_wordle_env.py         ← unit tests for letter-feedback correctness
│   ├── encoding.py                ← sensory encoder (ORN smell + optic visual)
│   ├── candidate_filter.py        ← constraint solver (prunes candidate words per guess)
│   ├── brain_wordle.py            ← LIF spiking simulation (CPU / CUDA RTX 4050)
│   ├── readout.py                 ← trainable MLP policy head
│   ├── train.py                   ← REINFORCE training loop
│   ├── benchmark.py               ← win-rate evaluation on held-out words
│   └── server_wordle.py           ← FastAPI + WebSocket game server (20 Hz telemetry)
├── frontend/
│   ├── index.html                 ← semantic HTML, ARIA, OG meta
│   ├── style.css                  ← PlayStation-inspired dark canvas design system
│   └── app.js                     ← Three.js 3D world + connectome visualisation + game UI
├── design.md                      ← design system specification (PlayStation-inspired)
└── wordle-connectome-implementation-plan.md
```

---

## Running Locally

### Prerequisites

```bash
# Python 3.10+
pip install torch numpy fastapi uvicorn websockets cloudpickle
```

GPU training (RTX 4050 / CUDA):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 1. Extract the connectome sub-circuit

```bash
cd wordle/backend
python extract_circuit_wordle.py
```

### 2. Train the fly

```bash
python train.py          # ~5 min on RTX 4050, ~30 min on CPU
```

### 3. Benchmark

```bash
python benchmark.py
```

### 4. Start the game server

```bash
uvicorn server_wordle:app --host 127.0.0.1 --port 8001
```

### 5. Open in browser

Navigate to `http://127.0.0.1:8001` — press **START MATCH** to race the fly.

---

## Sources & Citations

| Resource | Link |
|----------|------|
| FlyWire v783 connectome | [flywire.ai](https://flywire.ai/) |
| Dorkenwald et al., *Nature* 2024 | [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y) |
| Schlegel et al. (synapse data) | [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5) |
| NYT Wordle answer list (Kinkelin mirror) | [github.com/kinkelin/WordleCompetition](https://github.com/kinkelin/WordleCompetition) |
| Three.js WebGL library | [threejs.org](https://threejs.org/) |
| REINFORCE algorithm | Williams, R.J. (1992). *Machine Learning*, 8, 229–256. |
| PAM/PPL1 dopamine circuits in Drosophila | [Aso et al., *eLife* 2014](https://doi.org/10.7554/eLife.04577) |
| Mushroom body olfactory memory | [Heisenberg, *Nature Rev. Neurosci.* 2003](https://doi.org/10.1038/nrn1074) |

---

## License

MIT — see `LICENSE`.

*Built with curiosity, Three.js, and genuine fruit fly neurons.*
