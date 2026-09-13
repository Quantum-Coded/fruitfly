"""
train.py
Reinforcement learning training loop for FlyWordle.
Uses dopamine-gated REINFORCE policy gradient to train the decision readout layer
while keeping the real 3,127-neuron FlyWire connectome frozen.
"""

import os
import time
import random
import torch
import torch.optim as optim
import numpy as np

from wordle_env import WordleEnv
from brain_wordle import WordleBrain
from encoding import WordleSensoryEncoder
from candidate_filter import CandidateFilter
from readout import WordleReadout

def train(episodes: int = 1500, lr: float = 1e-3, save_path: str = None):
    base_dir = os.path.dirname(__file__)
    if save_path is None:
        save_path = os.path.join(base_dir, 'readout_weights.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Training FlyWordle on {device.upper()} ---")

    env = WordleEnv()
    candidate_filter = CandidateFilter(env.answers)
    circuit_data = np.load(os.path.join(base_dir, 'data', 'circuit_wordle.npz'), allow_pickle=True)
    encoder = WordleSensoryEncoder(circuit_data, device=device)
    brain = WordleBrain(device=device)

    readout = WordleReadout(n_descending=len(brain.motor_idx)).to(device)
    optimizer = optim.Adam(readout.parameters(), lr=lr)

    print(f"Answer dictionary size: {len(env.answers)} words.")
    print(f"Beginning training for {episodes} episodes...")

    win_history = []
    guesses_history = []
    t0 = time.time()

    for ep in range(1, episodes + 1):
        secret = env.reset()
        brain.reset_state()

        log_probs = []
        last_guess = None
        last_fb = None

        for turn in range(env.max_guesses):
            # 1. Sensory encoding
            drive = encoder.encode_guess_and_feedback(guess=last_guess, feedback=last_fb)

            # 2. Connectome LIF simulation pass
            active_idx, telemetry = brain.step(drive, n_substeps=20)

            # 3. Candidate shortlist from constraints
            candidates = candidate_filter.filter_words(env.guesses, env.feedbacks, top_k=60)
            if not candidates:
                candidates = random.sample(env.answers, min(30, len(env.answers)))

            # 4. Readout scores candidates
            desc_rates = brain.get_descending_rates()
            probs, logits = readout.score_candidates(desc_rates, candidates)

            # Sample guess from distribution (exploration)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            guess = candidates[action_idx.item()]
            log_prob = dist.log_prob(action_idx)
            log_probs.append(log_prob)

            # 5. Step game environment
            feedback, done, guesses_used, won = env.step(guess)
            last_guess = guess
            last_fb = feedback

            if done:
                break

        # 6. Reward shaping & Dopamine gate
        if won:
            # Reward: 1.0 base + bonus for fewer guesses
            reward = 1.0 + (6 - guesses_used) * 0.25
            dopamine_pulse = reward
            win_history.append(1)
            guesses_history.append(guesses_used)
        else:
            reward = -1.0
            dopamine_pulse = 0.0
            win_history.append(0)
            guesses_history.append(6)

        # Pulse dopamine in connectome brain
        da_drive = encoder.encode_dopamine_reward(dopamine_pulse)
        brain.step(da_drive, n_substeps=25)

        # 7. Policy gradient update (REINFORCE)
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for lp in log_probs:
            loss = loss - lp * reward
        loss.backward()
        optimizer.step()

        # Logging
        if ep % 100 == 0 or ep == episodes:
            recent_wins = win_history[-100:]
            recent_guesses = guesses_history[-100:]
            win_rate = (sum(recent_wins) / len(recent_wins)) * 100
            avg_guesses = sum(recent_guesses) / len(recent_guesses)
            elapsed = time.time() - t0
            print(f"Episode {ep:4d}/{episodes} | Win Rate (last 100): {win_rate:5.1f}% | Avg Guesses: {avg_guesses:.2f} | Speed: {ep/elapsed:.1f} ep/s")

    # Save trained readout weights
    torch.save(readout.state_dict(), save_path)
    print(f"Trained readout weights saved to {save_path}!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(episodes=args.episodes, lr=args.lr)
