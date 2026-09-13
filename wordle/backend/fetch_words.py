"""
fetch_words.py
Downloads and caches the official Wordle 5-letter word lists:
- words_answers.txt: The ~2,315 curated official answer words.
- words_valid.txt: The ~14,855 total valid guess words.
"""

import os
import urllib.request

def fetch_word_lists():
    words_dir = os.path.join(os.path.dirname(__file__), 'words')
    os.makedirs(words_dir, exist_ok=True)
    answers_path = os.path.join(words_dir, 'words_answers.txt')
    valid_path = os.path.join(words_dir, 'words_valid.txt')

    print("Fetching official Wordle answer list...")
    answers_url = 'https://raw.githubusercontent.com/Kinkelin/WordleCompetition/main/data/official/shuffled_real_wordles.txt'
    req = urllib.request.Request(answers_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            lines = resp.read().decode('utf-8', errors='ignore').splitlines()
            answers = [
                w.strip().upper() for w in lines
                if w.strip() and not w.strip().startswith('#') and len(w.strip()) == 5 and w.strip().isalpha()
            ]
    except Exception as e:
        print(f"Error fetching official answers, falling back to backup: {e}")
        # Fallback list of common words
        fallback_url = 'https://raw.githubusercontent.com/charlesreid1/five-letter-words/master/sgb-words.txt'
        with urllib.request.urlopen(fallback_url) as resp:
            lines = resp.read().decode('utf-8', errors='ignore').splitlines()
            answers = [w.strip().upper() for w in lines if len(w.strip()) == 5 and w.strip().isalpha()][:2315]

    answers = sorted(list(set(answers)))
    with open(answers_path, 'w', encoding='utf-8') as f:
        for w in answers:
            f.write(w + '\n')
    print(f"Saved {len(answers)} answer words to {answers_path}")

    print("Fetching full valid Wordle guess list...")
    valid_url = 'https://raw.githubusercontent.com/tabatkins/wordle-list/main/words'
    req2 = urllib.request.Request(valid_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req2) as resp:
        lines = resp.read().decode('utf-8', errors='ignore').splitlines()
        valid = [
            w.strip().upper() for w in lines
            if len(w.strip()) == 5 and w.strip().isalpha()
        ]

    valid = sorted(list(set(valid + answers)))
    with open(valid_path, 'w', encoding='utf-8') as f:
        for w in valid:
            f.write(w + '\n')
    print(f"Saved {len(valid)} total valid guess words to {valid_path}")

if __name__ == '__main__':
    fetch_word_lists()
