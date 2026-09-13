"""
test_wordle_env.py
Unit tests for Wordle environment logic and edge cases.
"""

from wordle_env import WordleEnv

def test_wordle_feedback():
    # Exact match
    fb = WordleEnv.evaluate_guess("CRANE", "CRANE")
    assert fb == [2, 2, 2, 2, 2], f"Expected all green, got {fb}"

    # No match
    fb = WordleEnv.evaluate_guess("PLUMB", "FIGHT")
    assert fb == [0, 0, 0, 0, 0], f"Expected all gray, got {fb}"

    # Duplicate letters: ABBEY vs KEEPS -> only first E is yellow
    fb = WordleEnv.evaluate_guess("KEEPS", "ABBEY")
    assert fb == [0, 1, 0, 0, 0], f"Expected [0, 1, 0, 0, 0], got {fb}"

    # Secret has 2 E's, Guess has 2 E's: both should be yellow [1, 0, 1, 1, 0]
    fb = WordleEnv.evaluate_guess("SPEED", "ERASE")
    assert fb == [1, 0, 1, 1, 0], f"Expected [1, 0, 1, 1, 0], got {fb}"
    # Let's check ERASE vs SPEED:
    # Secret = SPEED (S:1, P:1, E:2, D:1)
    # Guess = ERASE:
    # E pos 0 -> yellow (1)
    # R pos 1 -> gray (0)
    # A pos 2 -> gray (0)
    # S pos 3 -> yellow (1)
    # E pos 4 -> yellow (1)
    fb = WordleEnv.evaluate_guess("ERASE", "SPEED")
    assert fb == [1, 0, 0, 1, 1], f"Expected [1, 0, 0, 1, 1], got {fb}"

    # Green takes priority over yellow for duplicate letters
    # Secret: SLATE, Guess: ALLOY -> First A is yellow, Second letter L is green
    fb = WordleEnv.evaluate_guess("ALLOY", "SLATE")
    assert fb == [1, 2, 0, 0, 0], f"Expected [1, 2, 0, 0, 0], got {fb}"

    print("All Wordle feedback unit tests passed successfully!")

def test_wordle_episode():
    env = WordleEnv()
    secret = env.reset("CRANE")
    assert secret == "CRANE"

    fb, done, count, won = env.step("SLATE")
    assert count == 1
    assert not done
    assert not won
    assert fb == [0, 0, 2, 0, 2] # A and E are green

    fb, done, count, won = env.step("CRANE")
    assert count == 2
    assert done
    assert won
    assert fb == [2, 2, 2, 2, 2]

    print("Wordle episode tests passed successfully!")

if __name__ == '__main__':
    test_wordle_feedback()
    test_wordle_episode()
