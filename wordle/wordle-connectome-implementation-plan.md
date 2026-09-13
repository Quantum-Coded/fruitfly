# Implementation plan: real fly-brain Wordle solver (RL + dopamine reward + 3D embodiment)

## 0. Architecture recap

- **Decision**: real FlyWire connectome (frozen) + a small trained readout, picks which word to guess
- **Senses**: real visual neurons encode tile position/color, real olfactory receptor neurons encode letter identity
- **Learning**: reinforcement learning — no labeled dataset, the readout improves purely from playing
- **Reward**: real dopaminergic neurons (PAM/PPL1, 331 confirmed in the data) fire when a guess is fully correct, and that signal drives the update
- **Body**: a MuJoCo biomechanical fly (NeuroMechFly v2) physically walks to letter tiles and taps them, for the visual — kept separate from the decision logic

This plan has 8 phases. Each has concrete steps, a deliverable, and where any external data comes from.

---

## Data sources (all of them, up front)

| What | Source | Notes |
|---|---|---|
| Full FlyWire v783 connectome (139k neurons, 15M synapses, 3D coordinates, cell types) | `github.com/erojasoficial-byte/fly-brain` (already cloned locally) | Open, MIT-licensed mirror of real FlyWire data |
| Dopaminergic neuron identities (PAM01–15, PPL1) | Same connectome annotation file | Already confirmed present: 331 real neurons |
| NeuroMechFly v2 biomechanical fly body + MuJoCo assets | Same repo (`fly-brain` bundles it) | Open source |
| Valid Wordle guess list (~10,657 words) | GitHub mirror, e.g. `tabatkins/wordle-list` | Public, free, pulled once |
| Official Wordle answer list (~2,315 words) | GitHub/gist mirrors of the original NYT list | Public, free, pulled once. If a mirror is unavailable, fall back to a filtered common-5-letter-word list (e.g. `dwyl/english-words`) as a substitute answer pool |
| Training games | **Generated, not downloaded** | RL produces its own practice data by playing against the real answer list — no pre-existing game dataset needed |

No personal, scraped, or proprietary data anywhere in this project.

---

## Phase 1 — Extract the real Wordle-relevant circuit

**Goal:** a real sub-circuit covering senses, memory/learning center, reward, and motor output.

1.1. Identify real neuron populations by `cell_type` in the connectome annotations:
   - Visual/optic-lobe neurons — for the 5 tile "pixels"
   - 26 distinct real olfactory receptor (ORN) classes — one assigned per letter of the alphabet
   - Mushroom body Kenyon cells + mushroom body output neurons (MBONs) — the learning center
   - PAM (all 15 subtypes) + PPL1 dopaminergic neurons — the reward pathway
   - Descending/motor neurons — the decision readout

1.2. Extend `extract_circuit.py` (already built for the odor project) to pull these populations plus their real induced synaptic subgraph (reuse the same BFS + induced-subgraph approach).

1.3. Save as `circuit_wordle.npz` (ids, 3D coords, cell types, signed weights).

**Deliverable:** real sub-circuit file, neuron counts per population logged and sanity-checked (expect low thousands, similar scale to the odor project's 3,127).

---

## Phase 2 — Classical Wordle engine (ground truth, no ML yet)

2.1. Pull the two real word lists (see data sources table); clean/dedupe/lowercase.

2.2. Implement the feedback function exactly matching real Wordle rules, including correct handling of repeated letters (this is the most common bug source — test against known real examples).

2.3. Implement an episode class: `reset(secret_word)`, `step(guess) -> (feedback, done, guesses_used)`.

**Deliverable:** `wordle_env.py`, unit-tested against real known Wordle results.

---

## Phase 3 — Sensory encoding layer

3.1. Fixed mapping: tile position (1–5) × color (green/yellow/gray) → injected current pattern across the real visual neurons, placed retinotopically using their real coordinates.

3.2. Fixed mapping: letter (A–Z) → injected current on that letter's assigned real ORN population.

3.3. Combine into one function: `encode_state(guess, feedback) -> injection_vector`.

**Deliverable:** `encoding.py`.

---

## Phase 4 — Brain forward pass + readout

4.1. Reuse the existing LIF engine (`brain.py` from the odor project), pointed at `circuit_wordle.npz`.

4.2. Readout: firing rate of descending/motor neurons → a scalar score for one candidate guess.

4.3. Candidate shortlist generator — a cheap classical pre-filter (e.g. top ~100 words by letter-frequency heuristic) so the connectome only has to be evaluated on a bounded shortlist each turn, not all ~10,657 valid words.

**Deliverable:** `readout.py`, `candidate_filter.py`.

---

## Phase 5 — Dopamine-gated RL training loop

5.1. Reward shaping: large positive reward if solved (bigger reward for solving in fewer guesses), penalty if failed after 6 guesses.

5.2. On a correct final guess, inject a reward-proportional current into the real PAM/PPL1 neurons — this is both the visual "reward pulse" and, mechanically, the signal that gates the learning update.

5.3. Policy gradient (REINFORCE): treat readout scores as logits over the shortlist, sample the guess actually played, and after the episode ends, push the small trained readout's weights toward whatever it did in episodes that scored well. The 138k-neuron connectome itself is never updated — only this small layer.

5.4. *(Optional, more advanced — do after 5.1–5.3 work)*: implement the literal three-factor Hebbian rule real fly learning uses — Kenyon cell → MBON synaptic weight changes gated by the dopamine signal — as a second, more biologically faithful training pathway to compare against plain policy gradient.

5.5. Training loop: each episode, sample a real secret word from the answer list, play up to 6 guesses, collect the trajectory, apply the update. Start with a few thousand episodes to confirm learning is happening (rising average reward / falling guesses-to-solve) before scaling up.

**Deliverable:** `train.py`, saved readout checkpoints, a training curve.

---

## Phase 6 — Benchmark

6.1. Freeze the trained readout. Play every word in the real official answer list once (or a large random sample).

6.2. Compute: win rate, average guesses-to-solve, average information gain per guess.

6.3. Compare against: the mathematically optimal solver (known exact ceiling, ~3.4 guesses), random guessing, and a plain MLP trained via the identical RL loop but with the connectome removed entirely (fair "real brain vs standard AI" comparison).

**Deliverable:** `benchmark.py`, a results table/chart — this is your headline LinkedIn number.

---

## Phase 7 — MuJoCo embodiment and 3D animation

7.1. Build a flat "letter grid" MuJoCo scene: 26 letter tiles + an enter tile, laid out within reach of the fly body. Reuse the NeuroMechFly v2 fly model already bundled in the `fly-brain` repo — don't rebuild a body from scratch.

7.2. Scripted walk-to-tile-and-tap controller: given a target tile's (x, y), generate a walking path (reuse the same heading-based movement approach from the odor-navigation project) ending in a tap animation. This is scripted, not learned — see the earlier note on why full learned locomotion is a much harder, separate problem not worth taking on here.

7.3. Wire the pieces together: the decision layer (Phases 1–6) outputs a letter → the controller walks the fly there and taps it → repeat 5 times per guess → reveal the color feedback on the grid tiles → loop.

7.4. Rendering: use MuJoCo's built-in renderer to export frames/video. Optionally mirror the real-time "live brain activity" panel from the odor-navigation project (the neuron point cloud that lights up per spike) alongside the grid view, synced to the same timeline — this is the same visual language as your friend's chess post.

**Deliverable:** `mujoco_env.py`, `walk_controller.py`, a rendered demo clip.

---

## Phase 8 — Package for the post

8.1. Record one full, clean, solved game end-to-end: physical walk/tap animation + live brain activity panel + running stats.

8.2. Export as video/GIF.

8.3. Overlay the Phase 6 benchmark numbers (avg guesses, win rate, vs. optimal solver, vs. plain AI) in the same "stats bar" style as the chess post you showed me.

**Deliverable:** final video asset, ready to post.

---

## Honest notes on scope

- Phase 5.4 (the literal three-factor dopamine plasticity rule) is the most scientifically ambitious piece and the most likely to need iteration — treat 5.1–5.3 as the working baseline and 5.4 as a stretch upgrade once that's solid.
- Phase 7's walking is scripted, not RL-learned — flagging this now so it's a deliberate choice, not a surprise later.
- Everything through Phase 6 can be validated on a laptop; Phase 7 (MuJoCo rendering) is the more hardware/time-intensive phase — budget accordingly.
