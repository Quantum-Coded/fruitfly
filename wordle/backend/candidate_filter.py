"""
candidate_filter.py
Classical constraint solver and candidate shortlist generator.

Maintains Wordle constraints across guesses:
  - Exact green matches: letter must be at exact position
  - Misplaced yellow matches: letter must be present in word, but NOT at that position
  - Absent gray matches: letter cannot appear (accounting for duplicate letter rules)

Ranks consistent words by letter frequency / positional entropy and returns
the top-K shortlist for brain evaluation.
"""

from collections import Counter
from typing import List, Set, Dict, Optional, Tuple

# Precomputed English letter frequencies (Wordle distribution)
LETTER_FREQ = {
    'E': 12.0, 'T': 9.0, 'A': 8.5, 'O': 7.5, 'I': 7.0, 'N': 6.7, 'S': 6.3,
    'H': 6.1, 'R': 6.0, 'D': 4.3, 'L': 4.0, 'C': 2.8, 'U': 2.8, 'M': 2.4,
    'W': 2.4, 'F': 2.2, 'G': 2.0, 'Y': 2.0, 'P': 1.9, 'B': 1.5, 'V': 1.0,
    'K': 0.8, 'J': 0.15, 'X': 0.15, 'Q': 0.1, 'Z': 0.07
}

class CandidateFilter:
    def __init__(self, dictionary: List[str]):
        self.all_words = [w.strip().upper() for w in dictionary if len(w.strip()) == 5]

    def filter_words(
        self,
        guesses: List[str],
        feedbacks: List[List[int]],
        top_k: int = 100
    ) -> List[str]:
        """
        Given the history of guesses and feedbacks, returns consistent candidate words
        ranked by letter frequency and diversity.
        """
        if not guesses:
            # First turn: return top opening words
            ranked = sorted(self.all_words, key=self._word_heuristic_score, reverse=True)
            return ranked[:top_k]

        # Build constraints from history
        # 1. exact_match[pos] = letter
        exact_matches: Dict[int, str] = {}
        # 2. forbidden_pos[letter] = set of positions where letter cannot be
        forbidden_pos: Dict[str, Set[int]] = {}
        # 3. min_counts[letter] = minimum times letter must appear
        min_counts: Dict[str, int] = {}
        # 4. max_counts[letter] = maximum times letter can appear
        max_counts: Dict[str, int] = {}

        for guess, feedback in zip(guesses, feedbacks):
            guess = guess.upper()
            turn_letter_green_or_yellow = Counter()
            turn_letter_gray = set()

            for pos in range(5):
                char = guess[pos]
                fb = feedback[pos]
                if fb == 2:  # Green
                    exact_matches[pos] = char
                    turn_letter_green_or_yellow[char] += 1
                elif fb == 1:  # Yellow
                    if char not in forbidden_pos:
                        forbidden_pos[char] = set()
                    forbidden_pos[char].add(pos)
                    turn_letter_green_or_yellow[char] += 1
                elif fb == 0:  # Gray
                    turn_letter_gray.add(char)
                    if char not in forbidden_pos:
                        forbidden_pos[char] = set()
                    forbidden_pos[char].add(pos)

            # Update min and max counts for this guess
            for char, count in turn_letter_green_or_yellow.items():
                min_counts[char] = max(min_counts.get(char, 0), count)

            # For letters that had grays in this guess:
            for char in turn_letter_gray:
                if char in turn_letter_green_or_yellow:
                    # Letter was present some number of times, but this extra instance was gray
                    max_counts[char] = turn_letter_green_or_yellow[char]
                else:
                    # Letter is completely absent from secret word
                    max_counts[char] = 0

        # Filter all words against collected constraints
        candidates = []
        for word in self.all_words:
            if self._satisfies_constraints(word, exact_matches, forbidden_pos, min_counts, max_counts):
                candidates.append(word)

        if not candidates:
            # Fallback if dictionary discrepancy
            return self.all_words[:top_k]

        # Rank candidates by letter frequency heuristic
        ranked = sorted(candidates, key=self._word_heuristic_score, reverse=True)
        return ranked[:top_k]

    def _satisfies_constraints(
        self,
        word: str,
        exact_matches: Dict[int, str],
        forbidden_pos: Dict[str, Set[int]],
        min_counts: Dict[str, int],
        max_counts: Dict[str, int]
    ) -> bool:
        # Check greens
        for pos, char in exact_matches.items():
            if word[pos] != char:
                return False

        # Check yellows / forbidden positions
        for char, positions in forbidden_pos.items():
            for pos in positions:
                if word[pos] == char:
                    return False

        # Check letter counts
        word_counts = Counter(word)
        for char, min_c in min_counts.items():
            if word_counts.get(char, 0) < min_c:
                return False

        for char, max_c in max_counts.items():
            if word_counts.get(char, 0) > max_c:
                return False

        return True

    def _word_heuristic_score(self, word: str) -> float:
        """Heuristic score favoring common unique letters (high information entropy)."""
        unique_letters = set(word)
        score = sum(LETTER_FREQ.get(c, 0.5) for c in unique_letters)
        # Bonus for having 5 distinct letters (avoids duplicates early on)
        if len(unique_letters) == 5:
            score += 4.0
        return score
