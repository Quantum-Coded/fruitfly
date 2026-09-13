"""
wordle_env.py
Classical Wordle game environment and evaluation engine.

Feedback values:
  0: Gray   (Letter not in target word, or excess occurrence)
  1: Yellow (Letter in target word, but wrong position)
  2: Green  (Letter in target word at this exact position)
"""

import os
import random
from typing import List, Tuple, Optional, Dict

class WordleEnv:
    def __init__(self, answers_file: Optional[str] = None, valid_file: Optional[str] = None):
        base_dir = os.path.dirname(__file__)
        if answers_file is None:
            answers_file = os.path.join(base_dir, 'words', 'words_answers.txt')
        if valid_file is None:
            valid_file = os.path.join(base_dir, 'words', 'words_valid.txt')

        with open(answers_file, 'r', encoding='utf-8') as f:
            self.answers = [w.strip().upper() for w in f if w.strip()]

        with open(valid_file, 'r', encoding='utf-8') as f:
            self.valid_words = set(w.strip().upper() for w in f if w.strip())

        self.valid_words.update(self.answers)
        self.secret_word: str = ""
        self.guesses: List[str] = []
        self.feedbacks: List[List[int]] = []
        self.max_guesses = 6
        self.done = False
        self.won = False

    def reset(self, secret_word: Optional[str] = None) -> str:
        if secret_word is not None:
            self.secret_word = secret_word.strip().upper()
        else:
            self.secret_word = random.choice(self.answers)

        self.guesses = []
        self.feedbacks = []
        self.done = False
        self.won = False
        return self.secret_word

    @staticmethod
    def evaluate_guess(guess: str, secret: str) -> List[int]:
        """
        Evaluates a 5-letter guess against secret according to standard Wordle rules.
        Correctly handles duplicate letters:
          - Pass 1: exact matches (Green = 2)
          - Pass 2: misplaced matches (Yellow = 1) from remaining letter pool
          - Pass 3: remaining (Gray = 0)
        """
        guess = guess.upper()
        secret = secret.upper()
        assert len(guess) == 5 and len(secret) == 5

        feedback = [0] * 5
        secret_pool: Dict[str, int] = {}

        # Count frequencies in secret
        for c in secret:
            secret_pool[c] = secret_pool.get(c, 0) + 1

        # Pass 1: Identify all greens (correct letter & position)
        for i in range(5):
            if guess[i] == secret[i]:
                feedback[i] = 2
                secret_pool[guess[i]] -= 1

        # Pass 2: Identify yellows (correct letter, wrong position, if available in pool)
        for i in range(5):
            if feedback[i] != 2:
                c = guess[i]
                if secret_pool.get(c, 0) > 0:
                    feedback[i] = 1
                    secret_pool[c] -= 1

        return feedback

    def is_valid_guess(self, word: str) -> bool:
        return len(word) == 5 and word.upper() in self.valid_words

    def step(self, guess: str) -> Tuple[List[int], bool, int, bool]:
        """
        Plays a guess.
        Returns: (feedback, done, guesses_used, won)
        """
        guess = guess.strip().upper()
        if self.done:
            raise ValueError("Episode is already finished. Call reset() to start a new game.")
        if len(guess) != 5:
            raise ValueError(f"Guess must be 5 letters, got: '{guess}'")

        feedback = self.evaluate_guess(guess, self.secret_word)
        self.guesses.append(guess)
        self.feedbacks.append(feedback)

        if guess == self.secret_word:
            self.done = True
            self.won = True
        elif len(self.guesses) >= self.max_guesses:
            self.done = True
            self.won = False

        return feedback, self.done, len(self.guesses), self.won

    def get_state(self) -> Dict:
        return {
            "secret_word": self.secret_word if self.done else None,
            "guesses": self.guesses,
            "feedbacks": self.feedbacks,
            "guesses_used": len(self.guesses),
            "max_guesses": self.max_guesses,
            "done": self.done,
            "won": self.won,
        }
